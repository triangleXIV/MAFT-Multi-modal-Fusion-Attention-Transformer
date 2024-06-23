# MFAT：Multi-modal Fusion Attention Transformer

你可以选择官方所提供的数据集和预训练模型，或者用我上传的实际使用的数据集和预训练模型：

You can choose the dataset and pre-trained models provided by the official sources, or use the dataset and pre-trained models that I have uploaded:

<br><br>
官方渠道(offical)：

数据集下载(Twitter15/17)见如下链接(twitterdatasets)  

The dataset used can be downloaded from:  (twitterdatasets)  

https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view

<br><br>
所使用的预训练模型有两个(根据自己的配置在此选择相应的预训练模型和config文件)  (pretrains)

The pre-trained model used can be downloaded from(Choose the appropriate pre-trained model and config file according to your configuration here):  (pretrains)

https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file  

https://github.com/microsoft/DeBERTa  

<br><br>
本论文实际使用的数据集与预训练模型（123pan与onedrive任选其一）：

The dataset and pre-trained models actually used in this paper（Choose either 123pan or OneDrive）:

[123pan](https://www.123pan.com/s/f3giVv-JKS3H.html)||[onedrive](https://1drv.ms/f/s!Akl56EV1csnmokSrk4mguoLljFqN?e=G060F7)

<br><br>
运行train.py开始模型的训练，在训练开始前，请保证模型满足如下结构：

Run train.py to start the model training. Before starting the training, please ensure that the model meets the following structure:
```
├─.idea
│  └─inspectionProfiles
├─models
│  ├─deberta
│  ├─swin
├─output
├─pretrains
├─twitterdataset
│  ├─absa_data
│  │  ├─twitter
│  │  └─twitter2015
│  └─img_data
│      ├─twitter2015_images
│      └─twitter2017_images
```
