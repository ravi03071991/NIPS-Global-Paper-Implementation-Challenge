## Selective Classification for Deep Neural Networks

### Introduction

This is implementation of NIPS 2017 Paper [Selective Classification For Deep Neural Networks](https://arxiv.org/abs/1705.08500) as a part of [NIPS Global Paper Implementation Challenge](https://nurture.ai/)

### Dependencies

1. Python 3.6
2. Keras
3. Tensoflow

### Files needed to evaluate

- ImageNet validation dataset can be downloaded from [here](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads).
- ILSVRC2012_validation_ground_truth.txt contains ground truth labels for ImageNet validation dataset.
- imagenet-classes-dict.dat is a pickle dictionary, if you input a class you get a number from 1 to 1000 corresponding to the ground truth   in the ILSVRC2012_validation_ground_truth.txt file.
- The weights of the model trained as suggested in paper on CIFAR-10 and CIFAR-100 datasets can be downloaded from [CIFAR-10 WEIGHTS](https://drive.google.com/open?id=14L6j0jtNDibhKTtMdr6OtHCV_wVonRtu)(93.67% accuracy)and [CIFAR-100 WEIGHTS](https://drive.google.com/open?id=19n10aUsTc8vxUCHFw_wfRvCt-L1vhQfG)(70.52% accuracy).

### Evaluation

- Evaluating on CIFAR-10 dataset.
```
python eval/cifar10_vgg16.py
```
- Evaluating on CIFAR-100 dataset.
```
python eval/cifar100_vgg16.py
```
- Evaluating on ImageNet validation dataset using VGG16 top1.
```
python eval/vgg16_top1.py
```
- Evaluating on ImageNet validation dataset using VGG16 top5.
```
python eval/vgg16_top5.py
```
- Evaluating on ImageNet validation dataset using ResNet50 top1.
```
python eval/resnet50_top1.py
```
- Evaluating on ImageNet validation dataset using ResNet50 top5.
```
python eval/resnet50_top5.py
```
### Experiment Results

- On CIFAR-10 using VGG16.

| Desired Risk | Train Risk | Train Coverage | Test Risk | Test Coverage | Risk Bound|
|    :---:     |    :---:   |    :---:       |   :---:   |   :---:       |  :---:    |
|0.01|0.0039|0.7044|0.0046|0.6964|0.0093|   
|0.02|0.0121|0.8410|0.0140|0.8376|0.0199|   
|0.03|0.0207|0.8896|0.0226|0.8868|0.0299|   
|0.04|0.0294|0.9198|0.0293|0.9200|0.0399|   
|0.05|0.0382|0.9482|0.0388|0.9492|0.0498|   
|0.06|0.0473|0.9688|0.0477|0.9728|0.0599|  


- On CIFAR-100 dataset using VGG16.

| Desired Risk | Train Risk | Train Coverage | Test Risk | Test Coverage | Risk Bound|
|    :---:     |    :---:   |    :---:       |   :---:   |   :---:       |  :---:    |
|0.02|0.0031|0.1288|0.0074|0.1354|0.0185|   
|0.05|0.0319|0.4012|0.0344|0.4016|0.0488|   
|0.10|0.0792|0.5584|0.0821|0.5646|0.0099|   
|0.15|0.1268|0.6642|0.1279|0.6734|0.0149|   
|0.20|0.1756|0.7698|0.1746|0.7672|0.0199|   
|0.25|0.2253|0.8692|0.2263|0.8704|0.2499| 

- On ImageNet Validation dataset using VGG16 Top1.

| Desired Risk | Train Risk | Train Coverage | Test Risk | Test Coverage | Risk Bound|
|    :---:     |    :---:   |    :---:       |   :---:   |   :---:       |  :---:    |
|0.02|0.0118|0.1619|0.1011|0.1582|0.0198|
|0.05|0.0418|0.4084|0.0429|0.4052|0.0498|
|0.10|0.0904|0.5608|0.0926|0.5660|0.0999|
|0.15|0.1395|0.6741|0.1373|0.6762|0.1499|
|0.20|0.1891|0.7762|0.1855|0.7817|0.1999|
|0.25|0.2388|0.8736|0.2337|0.8770|0.2499|

- On ImageNet Validation dataset using VGG16 Top5.

| Desired Risk | Train Risk | Train Coverage | Test Risk | Test Coverage | Risk Bound|
|    :---:     |    :---:   |    :---:       |   :---:   |   :---:       |  :---:    |
|0.01|0.0055|0.2556|0.0071|0.2534|0.0099|
|0.02|0.0152|0.4798|0.0176|0.4823|0.0199|
|0.03|0.0247|0.5870|0.0254|0.5929|0.0299|
|0.04|0.0343|0.6763|0.0341|0.6785|0.0399|
|0.05|0.0440|0.7589|0.0414|0.7646|0.0499|
|0.06|0.0537|0.8148|0.0521|0.8196|0.0599|
|0.07|0.0634|0.8654|0.0622|0.8681|0.0699|

- On ImageNet Validation dataset using ResNet50 Top1.

| Desired Risk | Train Risk | Train Coverage | Test Risk | Test Coverage | Risk Bound|
|    :---:     |    :---:   |    :---:       |   :---:   |   :---:       |  :---:    |
|0.02|0.0122|0.1733|0.0114|0.1722|0.0199|  
|0.05|0.0422|0.4461|0.0455|0.4425|0.0499|
|0.10|0.0908|0.6141|0.0903|0.6156|0.0999|   
|0.15|0.1399|0.7336|0.1374|0.7328|0.1499|   
|0.20|0.1895|0.8438|0.1901|0.8458|0.1999|   
|0.25|0.2392|0.9381|0.2389|0.9386|0.2499|   

- On ImageNet Validation dataset using ResNet50 Top5.

| Desired Risk | Train Risk | Train Coverage | Test Risk | Test Coverage | Risk Bound|
|    :---:     |    :---:   |    :---:       |   :---:   |   :---:       |  :---:    |
|0.01|0.0053|0.2398|0.0062|0.2374|0.0999|   
|0.02|0.0153|0.4965|0.0156|0.4984|0.0199|   
|0.03|0.0249|0.6306|0.0236|0.6324|0.0299|   
|0.04|0.0346|0.7374|0.0321|0.7370|0.0399|   
|0.05|0.0442|0.8138|0.0408|0.8153|0.0499|   
|0.06|0.0539|0.8710|0.0501|0.8714|0.0599|   
|0.07|0.0636|0.9205|0.0622|0.9223|0.0699| 

### Notes on Experiments:

- Achieved 60% test coverage guaranteed with 99.9% probability at 3% error rate top-5 ImageNet classification.

