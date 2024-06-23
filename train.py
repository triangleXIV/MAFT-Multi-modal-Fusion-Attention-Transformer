import os
import argparse
import torch
import numpy as np
from torch.optim import AdamW
from yacs.config import CfgNode as CN
from tqdm import tqdm,trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from toolcls import AbmsaProcessor,seed_everything,convert_mm_examples_to_features,BertConfig
from models.swin.swintransformer import SwinTransformer,get_config
from sklearn.metrics import precision_recall_fscore_support
from models.deberta.spm_tokenizer import SPMTokenizer
from models.deberta.deberta import MFAT
from models.logs import logger
from torch.optim.lr_scheduler import LambdaLR
import math

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_C = CN()
config = _C.clone()
config.LOCAL_RANK = -1

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return (1-x)/(1-warmup)


def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    #f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--data_dir",default='./twitterdataset/absa_data/twitter',type=str,# twitter == twitter17  twitter15 == twitter15
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--img_ckpt", default='./pretrains/swin_base_patch4_window7_224_1k.pth', type=str,#path of img pretrain model
                    help="swin_base_patch4_window7_224_1k.pth, swin_small_patch4_window7_224_1k.pth , swin_tiny_patch4_window7_224_1k.pth ")
parser.add_argument("--spm_model_file",default='./pretrains/30k-clean.model',type=str)# tokenizer
parser.add_argument("--model_name_or_path", default='./pretrains', type=str,#text pretrain document
                    help="Path to pre-trained model or shortcut name selected in the list")
parser.add_argument("--task_name",default='twitter17',type=str,# twitter17 or twitter15
                    help="The name of the task to train.")
parser.add_argument("--output_dir",default='./output',type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument('--path_image', default='./twitterdataset/img_data/twitter2017_images',# dataset of img
                    help='path to images')
parser.add_argument('--cfg', type=str, default="./pretrains/swin_base_patch4_window7_224.yaml", metavar="FILE",
                    help='path to config file,swin_base_patch4_window7_224,swin_small_patch4_window7_224,swin_tiny_patch4_window7_224', )#img model config
parser.add_argument('--init_model',
                    type=str,
                    default='./pretrains/pytorch_model.bin',# weight of text model
                    help="The model state file used to initialize the model weights.pytorch_model.bin pytorch_model_small.bin pytorch_mode_tiny.bin")
parser.add_argument('--model_config',# config of MFAT
                    type=str,
                    default='./pretrains/config.json',
                    help="The config file of bert model.config.json config_small.json config_tiny.json")
parser.add_argument('--vocab_path',
                    default='./pretrains/spm.model',
                    type=str,
                    help="The path of the vocabulary")

## Other parameters
parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
parser.add_argument("--max_seq_length",default=64,type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--max_entity_length",default=16,type=int,
                    help="The maximum entity input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",action='store_true',default=True,
                    help="Whether to run training.")
parser.add_argument("--do_lower_case",action='store_true',default=True,
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--train_batch_size",default=16,type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",default=24,type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",default=3e-5,type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",default=8.0,type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",default=0.1,type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--no_cuda",action='store_true',default=False,
                    help="Whether not to use CUDA when available")
parser.add_argument("--local_rank",type=int,default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',type=int,default=4,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',type=int,default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',default=False,action='store_true',# only fp32
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale',type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--overwrite_output_dir', action='store_true',default=True,
                    help="Overwrite the content of the output directory")
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
args = parser.parse_args(args=[])

if args.task_name == "twitter17":
    args.path_image = "./twitterdataset/img_data/twitter2017_images"
elif args.task_name == "twitter15":
    args.path_image = "./twitterdataset/img_data/twitter2015_images"
else:
    print("The task name is not right!")
processors = {
        "twitter15": AbmsaProcessor,    # our twitter-2015 dataset
        "twitter17": AbmsaProcessor         # our twitter-2017 dataset
}
num_labels_task = {
    "twitter15": 3,                # our twitter-2015 dataset
    "twitter17": 3                     # our twitter-2017 dataset
}
seed_everything(args.seed)
task_name = args.task_name.lower()
#init output dir
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
args.output_dir = args.output_dir
if os.path.exists(args.output_dir) and os.listdir(
        args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
            args.output_dir))

if config.LOCAL_RANK == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(config.LOCAL_RANK)
    device = torch.device("cuda", config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1

args.device = device
processor = processors[task_name]()#read tsv
num_labels = num_labels_task[task_name]
label_list = processor.get_labels()
# load config
config_file = os.path.join(args.model_name_or_path, 'bert_config.json')# get
bert_config = BertConfig.from_json_file(config_file)
# load tokenizer
tokenizer = SPMTokenizer(args.vocab_path)

#create model and init the weight
model=MFAT(args, bert_config,args.init_model)
model.to(device)

#set img encoder
config = get_config(args)
encoder = SwinTransformer(img_size=224,
                          patch_size=config.MODEL.SWIN.PATCH_SIZE,
                          in_chans=config.MODEL.SWIN.IN_CHANS,
                          num_classes=config.MODEL.NUM_CLASSES,
                          embed_dim=config.MODEL.SWIN.EMBED_DIM,
                          depths=config.MODEL.SWIN.DEPTHS,
                          num_heads=config.MODEL.SWIN.NUM_HEADS,
                          window_size=config.MODEL.SWIN.WINDOW_SIZE,
                          mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                          qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                          qk_scale=config.MODEL.SWIN.QK_SCALE,
                          drop_rate=config.MODEL.DROP_RATE,
                          drop_path_rate=config.MODEL.DROP_PATH_RATE,
                          patch_norm=config.MODEL.SWIN.PATCH_NORM,
                          use_checkpoint=False)
pretrained_dict = torch.load(args.img_ckpt, map_location='cpu')
pretrained_dict = pretrained_dict['model']
unexpected_keys = {"head.weight", "head.bias"}
for key in unexpected_keys:
    del pretrained_dict[key]
missing_keys, unexpected_keys = encoder.load_state_dict(pretrained_dict, strict=False)
encoder.to(device)

train_examples = processor.get_train_examples(args.data_dir)
eval_examples = processor.get_dev_examples(args.data_dir)
num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)
t_total = num_train_steps
#set text optimizer param
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
# set img optimizer param
def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay,'weight_decay': 0.01},
            {'params': no_decay, 'weight_decay': 0.}]

skip = {'absolute_pos_embed'}
skip_keywords = {'relative_position_bias_table'}
optimizer_grouped_parameters2 = set_weight_decay(encoder, skip, skip_keywords)
# get parameters from text model parameters and img model parameters
optimizer_grouped_parameters = optimizer_grouped_parameters1 + optimizer_grouped_parameters2
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,eps=1e-6)

num_train_steps = int(
    len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
t_total = num_train_steps
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=int(t_total * args.warmup_proportion), t_total=t_total)
output_model_file = os.path.join(args.output_dir, "pytorch_model.pth")
output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.pth")
# train datasets to cuda
train_features = convert_mm_examples_to_features(
            train_examples, label_list, args.max_seq_length, args.max_entity_length, tokenizer, args.crop_size,
            args.path_image)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_s2_input_ids = torch.tensor([f.s2_input_ids for f in train_features], dtype=torch.long)
all_s2_input_mask = torch.tensor([f.s2_input_mask for f in train_features], dtype=torch.long)
all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in train_features], dtype=torch.long)
all_img_feats = torch.stack([f.img_feat for f in train_features])
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, \
                           all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,
                           all_img_feats, all_label_ids)
if args.local_rank == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                              drop_last=True)
# eval datasets to cuda
eval_features = convert_mm_examples_to_features(
    eval_examples, label_list, args.max_seq_length, args.max_entity_length, tokenizer, args.crop_size,
    args.path_image)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_s2_input_ids = torch.tensor([f.s2_input_ids for f in eval_features], dtype=torch.long)
all_s2_input_mask = torch.tensor([f.s2_input_mask for f in eval_features], dtype=torch.long)
all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in eval_features], dtype=torch.long)
all_img_feats = torch.stack([f.img_feat for f in eval_features])
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, \
                          all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids, \
                          all_img_feats, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)
global_step = 0
nb_tr_steps = 0
tr_loss = 0
max_acc = 0.0

logger.info("*************** Running training ***************")
for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
    logger.info("********** Epoch: " + str(train_idx) + " **********")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    model.train()
    encoder.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    progress_bar = tqdm(enumerate(train_dataloader), desc="Iteration", total=len(train_dataloader), position=0)
    for step, batch in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
            img_feats, label_ids = batch
        img_att = encoder(img_feats)
        loss = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, input_mask,
                         s2_input_mask, \
                         added_input_mask, label_ids)
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        scheduler.step()  # 使用学习率调整算法
        for param_group in optimizer.param_groups:
            progress_bar.set_description(f"Iteration (loss: {loss.item():.4f},lr:{param_group['lr']:.10f})")
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    logger.info("***** Running evaluation on Dev Set*****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    encoder.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    true_label_list = []
    pred_label_list = []
    #验证
    progress_bar = tqdm(eval_dataloader, desc="Evaluating", position=0)
    for input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
            img_feats, label_ids in progress_bar:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        added_input_mask = added_input_mask.to(device)
        segment_ids = segment_ids.to(device)
        s2_input_ids = s2_input_ids.to(device)
        s2_input_mask = s2_input_mask.to(device)
        s2_segment_ids = s2_segment_ids.to(device)
        img_feats = img_feats.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            img_att = encoder(img_feats)
            tmp_eval_loss = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids,
                                  input_mask, s2_input_mask, added_input_mask, label_ids)
            logits = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, input_mask,
                           s2_input_mask, added_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        true_label_list.append(label_ids)
        pred_label_list.append(logits)
        tmp_eval_accuracy = accuracy(logits, label_ids)
        progress_bar.set_description(f"Evaluating (tmp_eval_loss: {tmp_eval_loss.item():.4f})")
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    loss = tr_loss / nb_tr_steps if args.do_train else None
    true_label = np.concatenate(true_label_list)
    pred_outputs = np.concatenate(pred_label_list)
    precision, recall, F_score = macro_f1(true_label, pred_outputs)
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'f_score': F_score,
              'global_step': global_step,
              'loss': loss}
    logger.info("***** Dev Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    if eval_accuracy >= max_acc:
        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        encoder_to_save = encoder.module if hasattr(encoder,
                                                    'module') else encoder  # Only save the model it-self
        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(encoder_to_save.state_dict(), output_encoder_file)
        max_acc = eval_accuracy

# get the best weight to load , then put it to test datasets
torch.cuda.empty_cache()
model.load_state_dict(torch.load(output_model_file))
encoder.load_state_dict(torch.load(output_encoder_file))
eval_examples = processor.get_test_examples(args.data_dir)
eval_features = convert_mm_examples_to_features(
    eval_examples, label_list, args.max_seq_length, args.max_entity_length, tokenizer, args.crop_size,
    args.path_image)
logger.info("***** Running evaluation on Test Set*****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", args.eval_batch_size)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_s2_input_ids = torch.tensor([f.s2_input_ids for f in eval_features], dtype=torch.long)
all_s2_input_mask = torch.tensor([f.s2_input_mask for f in eval_features], dtype=torch.long)
all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in eval_features], dtype=torch.long)
all_img_feats = torch.stack([f.img_feat for f in eval_features])
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, \
                          all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,
                          all_img_feats, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
model.eval()
encoder.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
true_label_list = []
pred_label_list = []
for input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
        img_feats, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    added_input_mask = added_input_mask.to(device)
    segment_ids = segment_ids.to(device)
    s2_input_ids = s2_input_ids.to(device)
    s2_input_mask = s2_input_mask.to(device)
    s2_segment_ids = s2_segment_ids.to(device)
    img_feats = img_feats.to(device)
    label_ids = label_ids.to(device)
    with torch.no_grad():
        img_att = encoder(img_feats)
        tmp_eval_loss = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids,
                              input_mask, s2_input_mask, added_input_mask, label_ids)
        logits = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, input_mask,
                       s2_input_mask, added_input_mask)
    logits = logits.detach().cpu().numpy()
    label_ids = label_ids.to('cpu').numpy()
    true_label_list.append(label_ids)
    pred_label_list.append(logits)
    tmp_eval_accuracy = accuracy(logits, label_ids)
    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy
    nb_eval_examples += input_ids.size(0)
    nb_eval_steps += 1
eval_loss = eval_loss / nb_eval_steps
eval_accuracy = eval_accuracy / nb_eval_examples
loss = tr_loss / nb_tr_steps if args.do_train else None
true_label = np.concatenate(true_label_list)
pred_outputs = np.concatenate(pred_label_list)
precision, recall, F_score = macro_f1(true_label, pred_outputs)
result = {'eval_loss': eval_loss,
          'eval_accuracy': eval_accuracy,
          'precision': precision,
          'recall': recall,
          'f_score': F_score,
          'global_step': global_step,
          'loss': loss}
pred_label = np.argmax(pred_outputs, axis=-1)
fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')
for i in range(len(pred_label)):
    attstr = str(pred_label[i])
    fout_p.write(attstr + '\n')
for i in range(len(true_label)):
    attstr = str(true_label[i])
    fout_t.write(attstr + '\n')
output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
fout_p.close()
fout_t.close()
print(result)