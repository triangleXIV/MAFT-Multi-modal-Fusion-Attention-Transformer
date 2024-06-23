
from .config import ModelConfig
from .cache_utils import load_model_state
from .bert import *
from ..logs import logger
import math
import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from timm.models.layers import Mlp
import yaml

class NNModule(nn.Module):
    """ An abstract class to handle weights initialization and \
      a simple interface for dowloading and loading pretrained models.

    Args:

      config (:obj:`~DeBERTa.deberta.ModelConfig`): The model config to the module

    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def init_weights(self, module):
        """ Apply Gaussian(mean=0, std=`config.initializer_range`) initialization to the module.

        Args:

          module (:obj:`torch.nn.Module`): The module to apply the initialization.

        Example::

          class MyModule(NNModule):
            def __init__(self, config):
              # Add construction instructions
              self.bert = DeBERTa(config)

              # Add other modules
              ...

              # Apply initialization
              self.apply(self.init_weights)

        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def export_onnx(self, onnx_path, input):
        raise NotImplementedError

    @classmethod
    def load_model(cls, model_path, model_config=None, tag=None, no_cache=False, cache_dir=None, *inputs, **kwargs):
        """ Instantiate a sub-class of NNModule from a pre-trained model file.

        Args:

          model_path (:obj:`str`): Path or name of the pre-trained model which can be either,

            - The path of pre-trained model

            - The pre-trained DeBERTa model name in `DeBERTa GitHub releases <https://github.com/microsoft/DeBERTa/releases>`_, i.e. [**base, base_mnli, large, large_mnli**].

            If `model_path` is `None` or `-`, then the method will create a new sub-class without initialing from pre-trained models.

          model_config (:obj:`str`): The path of model config file. If it's `None`, then the method will try to find the the config in order:

            1. ['config'] in the model state dictionary.

            2. `model_config.json` aside the `model_path`.

            If it failed to find a config the method will fail.

          tag (:obj:`str`, optional): The release tag of DeBERTa, default: `None`.

          no_cache (:obj:`bool`, optional): Disable local cache of downloaded models, default: `False`.

          cache_dir (:obj:`str`, optional): The cache directory used to save the downloaded models, default: `None`. If it's `None`, then the models will be saved at `$HOME/.~DeBERTa`

        Return:

          :obj:`NNModule` : The sub-class object.

        """
        # Load config
        if model_config:
            config = ModelConfig.from_json_file(model_config)
        else:
            config = None
        model_config = None
        model_state = None
        if (model_path is not None) and (model_path.strip() == '-' or model_path.strip() == ''):
            model_path = None
        try:
            model_state, model_config = load_model_state(model_path, tag=tag, no_cache=no_cache, cache_dir=cache_dir)
        except Exception as exp:
            raise Exception(f'Failed to get model {model_path}. Exception: {exp}')

        if config is not None and model_config is not None:
            for k in config.__dict__:
                if k not in ['hidden_size',
                             'intermediate_size',
                             'num_attention_heads',
                             'num_hidden_layers',
                             'vocab_size',
                             'max_position_embeddings'] or (k not in model_config.__dict__) or (
                        model_config.__dict__[k] < 0):
                    model_config.__dict__[k] = config.__dict__[k]
        if model_config is not None:
            config = copy.copy(model_config)
        vocab_size = config.vocab_size
        # Instantiate model.
        model = DeBERTa(config, *inputs, **kwargs)
        if not model_state:
            return model
        # copy state_dict so _load_from_state_dict can modify it
        state_dict = model_state.copy()

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)

        def load(module, prefix='deberta.'):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        logger.warning(f'Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}, error_msgs: {error_msgs}')
        return model


class DeBERTa(torch.nn.Module):
    """ DeBERTa encoder
    This module is composed of the input embedding layer with stacked transformer layers with disentangled attention.

    Parameters:
      config:
        A model config class instance with the configuration to build a new model. The schema is similar to `BertConfig`, \
            for more details, please refer :class:`~DeBERTa.deberta.ModelConfig`

      pre_trained:
        The pre-trained DeBERTa model, it can be a physical path of a pre-trained DeBERTa model or a released configurations, \
            i.e. [**base, large, base_mnli, large_mnli**]

    """

    def __init__(self, config=None, pre_trained=None):
        super().__init__()
        state = None
        if pre_trained is not None:
            state, model_config = load_model_state(pre_trained)
            if config is not None and model_config is not None:
                for k in config.__dict__:
                    if k not in ['hidden_size',
                                 'intermediate_size',
                                 'num_attention_heads',
                                 'num_hidden_layers',
                                 'vocab_size',
                                 'max_position_embeddings']:
                        model_config.__dict__[k] = config.__dict__[k]
            config = copy.copy(model_config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.config = config
        self.pre_trained = pre_trained
        self.apply_state(state)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_all_encoded_layers=True,
                position_ids=None, return_att=False):
        """
        Args:
          input_ids:
            a torch.LongTensor of shape [batch_size, sequence_length] \
          with the word token indices in the vocabulary

          attention_mask:
            an optional parameter for input mask or attention mask.

            - If it's an input mask, then it will be torch.LongTensor of shape [batch_size, sequence_length] with indices \
          selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max \
          input sequence length in the current batch. It's the mask that we typically use for attention when \
          a batch has varying length sentences.

            - If it's an attention mask then it will be torch.LongTensor of shape [batch_size, sequence_length, sequence_length]. \
          In this case, it's a mask indicate which tokens in the sequence should be attended by other tokens in the sequence.

          token_type_ids:
            an optional torch.LongTensor of shape [batch_size, sequence_length] with the token \
          types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to \
          a `sentence B` token (see BERT paper for more details).

          output_all_encoded_layers:
            whether to output results of all encoder layers, default, True

        Returns:

          - The output of the stacked transformer layers if `output_all_encoded_layers=True`, else \
          the last layer of stacked transformer layers

          - Attention matrix of self-attention layers if `return_att=True`


        Example::

          # Batch of wordPiece token ids.
          # Each sample was padded with zero to the maxium length of the batch
          input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
          # Mask of valid input ids
          attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

          # DeBERTa model initialized with pretrained base model
          bert = DeBERTa(pre_trained='base')

          encoder_layers = bert(input_ids, attention_mask=attention_mask)

        """

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        ebd_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), position_ids,
                                     attention_mask)
        embedding_output = ebd_output['embeddings']
        encoder_output = self.encoder(embedding_output,
                                      attention_mask,
                                      output_all_encoded_layers=False, return_att=return_att)
        encoder_output.update(ebd_output)
        return encoder_output['hidden_states']

    def apply_state(self, state=None):
        """ Load state from previous loaded model state dictionary.

          Args:
            state (:obj:`dict`, optional): State dictionary as the state returned by torch.module.state_dict(), default: `None`. \
                If it's `None`, then will use the pre-trained state loaded via the constructor to re-initialize \
                the `DeBERTa` model
        """
        if self.pre_trained is None and state is None:
            return
        if state is None:
            state, config = load_model_state(self.pre_trained)
            self.config = config

        prefix = ''
        for k in state:
            if 'embeddings.' in k:
                if not k.startswith('embeddings.'):
                    prefix = k[:k.index('embeddings.')]
                break

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        self._load_from_state_dict(state, prefix=prefix, local_metadata=None, strict=True, missing_keys=missing_keys,
                                   unexpected_keys=unexpected_keys, error_msgs=error_msgs)


#这一块是自注意力融合多模态特征
class MultimodalEncoder(nn.Module):
    def __init__(self, config):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers[0]

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

#这一块是交叉注意力融合多模态特征
class BertCrossEncoder(nn.Module):
    def __init__(self, config):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers[0]

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output

class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}
AlbertLayerNorm = torch.nn.LayerNorm
BertLayerNorm = torch.nn.LayerNorm

class MFAT(nn.Module):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, args, bert_config,init_model):
        super(MFAT, self).__init__()
        self.num_labels = 3
        self.bert = NNModule.load_model(model_path=init_model, model_config=args.model_config)
        self.bert_s2= NNModule.load_model(model_path=init_model, model_config=args.model_config)

        with open(args.cfg, 'r') as file:
            yaml_content = yaml.safe_load(file)
        num_heads = yaml_content.get('MODEL', {}).get('SWIN', {}).get('NUM_HEADS', [])
        last_num_head = num_heads[-1] if num_heads else None
        self.img_hidden_state=last_num_head*32
        self.vismap2text = nn.Linear(self.img_hidden_state, bert_config.hidden_size)

        self.cross1 = BertCrossEncoder(bert_config)
        self.self1 = MultimodalEncoder(bert_config)
        self.cross2 = BertCrossEncoder(bert_config)
        self.self2 = MultimodalEncoder(bert_config)

        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.mlp1 = Mlp(in_features=bert_config.hidden_size, hidden_features=bert_config.hidden_size * 4, drop=0.1)
        self.norm1 = nn.LayerNorm(bert_config.hidden_size)
        self.mlp2 = Mlp(in_features=bert_config.hidden_size, hidden_features=bert_config.hidden_size * 4, drop=0.1)
        self.norm2 = nn.LayerNorm(bert_config.hidden_size)
        self.mlp3 = Mlp(in_features=bert_config.hidden_size, hidden_features=bert_config.hidden_size * 4, drop=0.1)
        self.norm3 = nn.LayerNorm(bert_config.hidden_size)
        self.mlp4 = Mlp(in_features=bert_config.hidden_size, hidden_features=bert_config.hidden_size * 4, drop=0.1)
        self.norm4 = nn.LayerNorm(bert_config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_config.hidden_size, 3)

    #这个输入类型应该和bert还是不一样的
    #def forward(self, input_ids, attention_mask=None, token_type_ids=None,
    #            position_ids=None, head_mask=None, labels=None):
    def forward(self, input_ids, s2_input_ids, visual_embeds_att, token_type_ids=None, s2_type_ids=None,
                attention_mask=None, s2_mask=None, added_attention_mask=None, labels=None, position_ids=None,head_mask=None):
        sequence_output = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)#none
        s2_output = self.bert_s2(s2_input_ids,
                                    attention_mask=s2_mask,
                                    token_type_ids=s2_type_ids,
                                    position_ids=position_ids)  # none
        #output返回的是 特征提取后的完整结果
        sequence_feature=sequence_output[0]
        s2_feature=s2_output[0]

        img_mask = added_attention_mask[:, :49]
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0
        converted_vis_embed_map = self.vismap2text(visual_embeds_att)

        extended_s2_mask = s2_mask.unsqueeze(1).unsqueeze(2)
        extended_s2_mask = extended_s2_mask.to(dtype=next(self.parameters()).dtype)
        extended_s2_mask = (1.0 - extended_s2_mask) * -10000.0
        s2_cross_output_layer = self.cross1(converted_vis_embed_map, s2_feature, extended_s2_mask)

        s2_cross_output_layer = s2_cross_output_layer + self.mlp1(self.norm1(s2_cross_output_layer))
        img_s2_cross_att = s2_cross_output_layer + self.self1(s2_cross_output_layer, extended_img_mask)
        img_s2_cross_att = img_s2_cross_att + self.mlp2(self.norm2(img_s2_cross_att))

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        seq_cross_output_layer = self.cross2(img_s2_cross_att, sequence_feature, extended_attention_mask)
        seq_cross_output_layer = seq_cross_output_layer + self.mlp3(self.norm3(seq_cross_output_layer))  # 残差连接 MLP3

        seq_img_s2_cross_att = seq_cross_output_layer + \
                               self.self2(seq_cross_output_layer, extended_img_mask)  # 自注意力去噪点
        seq_img_s2_cross_att = seq_img_s2_cross_att + self.mlp4(self.norm4(seq_img_s2_cross_att))  # 残差连接 MLP4

        seq_img_s2_cross_att = self.pool1(seq_img_s2_cross_att.transpose(1, 2))
        seq_img_s2_cross_att = torch.flatten(seq_img_s2_cross_att, 1)
        # seq_information = sequence_output[:, 0, :]  # 取出句子信息

        # pooled_output=self.norm2(torch.add(seq_img_s2_cross_att, seq_information))#逐元素累加法
        # pooled_output = torch.cat((seq_img_s2_cross_att, seq_information),dim=1)
        logits = self.classifier(seq_img_s2_cross_att)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
