# Defending Against Transfer-Based Adversarial Attacks Using SVD-Driven Feature Evolution

[//]: # ([Paper]&#40;&#41; )

> **Abstract:** Due to their high stealthiness and difficulty in detection, the transfer-based adversarial attacks pose 
> a significant challenge to the security and robustness of computer vision models. In this paper, we propose a 
> plug-and-play **_SVD-driven feature evolution module_** (**SDFEM**) to assist image classification 
> models in defending against transfer-based adversarial attacks. The SDFEM consists of "feature concatenation," 
> "feature reconstruction," and "feature weight optimization." After the adversarial examples are decomposed into 
> singular value features using Singular Value Decomposition (SVD), the above three components sequentially achieve 
> the concatenation of features along the channel dimension, the reconstruction of multi-level feature representations, 
> and the optimization of feature weights based on channel context, thereby suppressing the features that significantly 
> contribute to adversarial attacks. Extensive experiments demonstrate that the SDFEM effectively defends against 
> various types of transfer-based attacks, achieving state-of-the-art black-box robustness.

## Installation

```
conda create -n SDFEM python=3.11.9
conda activate SDFEM
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
PyTorch is not version-sensitive. The project can typically run on other versions of PyTorch as well. 
Furthermore, allow the system to automatically select the version when installing any other missing libraries.

## Training SDFEM

You can train the target model integrated with SDFEM.

* For SDFEM-WideResNet-28-10 on CIFAR-10
  `python train_sdfem.py --dataset_name cifar10 --model_name SDFEMWideResNet28x10 --pre_model_name ResNet18 --pre_model_load_path xxxx`

* For SDFEM-WideResNet-28-10 on Mini-ImageNet
  `python train_sdfem.py --dataset_name miniimagenet --model_name SDFEMWideResNet28x10 --pre_model_name ResNet18 --pre_model_load_path xxxx`


## Evaluating SDFEM

You can evaluate the SDFEM-WideResNet-28-10 model trained on the CIFAR-10 dataset using methods such as "Clean," "FGSM," "BIM," "DIM," "VMIM," and "VNIM."

The invocation method is as follows:  
  `python evaluate_sdfem.py --dataset_name cifar10 --model_name SDFEMWideResNet28x10 --model_load_path xxxx --sub_model_name xxxx --sub_model_load_path xxxx`
