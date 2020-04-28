# Sony AI Research Code

This repository contains code related to research papers in the area of 
Machine Learning and Artificial Intelligence, that have been published by Sony. 
We belief in transparent and reproducible research and therefore want to offer 
a quick and easy access to our findings. Hopefully, others will benefit as much
from them as we did.


## Available Code

### [**Mixed Precision DNNs: All you need is a good parametrization**](https://openreview.net/forum?id=Hyx0slrFvH&noteId=Hyx0slrFvH&invitationId=ICLR.cc/2020/Conference/Paper2519) ([Code](./mixed-precision-dnns))
> Uhlich, Stefan and Mauch, Lukas and Cardinaux, Fabien and Yoshiyama, Kazuki and Garcia, Javier Alonso and Tiedemann, Stephen and Kemp, Thomas and Nakamura, Akira.
> Published at the 8th International Conference on Learning Representations (ICLR) 2020
> *arXiv technical report ([arXiv 1905.11452]( https://arxiv.org/abs/1905.11452))*

![](mixed-precision-dnns/imgs/bitwidth.png)


Efficient deep neural network (DNN) inference on mobile or embedded devices typically 
involves quantization of the network parameters and activations. 
In particular, mixed precision networks achieve better performance than networks 
with homogeneous bitwidth for the same size constraint. Since choosing the optimal 
bitwidths is not straight forward, training methods, which can learn them, 
are desirable. Differentiable quantization with straight-through gradients allows 
to learn the quantizer's parameters using gradient methods. We show that a suited 
parametrization of the quantizer is the key to achieve a stable training and a good 
final performance. Specifically, we propose to parametrize the quantizer with the 
step size and dynamic range. The bitwidth can then be inferred from them. 
Other parametrizations, which explicitly use the bitwidth, consistently perform worse. 
We confirm our findings with experiments on CIFAR-10 and ImageNet and we obtain mixed 
precision DNNs with learned quantization parameters, achieving state-of-the-art performance. 

