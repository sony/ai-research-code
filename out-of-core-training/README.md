# Out-of-Core (OoC) Training

![](./imgs/overview.png)
![](./imgs/result.png)

This is the official implementation of [Out-of-core Training for Extremely Large-Scale Neural Networks With Adaptive Window-Based Scheduling](https://arxiv.org/abs/2010.14109).

We provide OoC feature as one of nnabla's utilities. You can enable OoC training on your nnabla script **with only a few additional lines**.
Please see [document](https://nnabla.readthedocs.io/en/latest/python/api/lms.html) for more detail information!

## Interactive Demo on Colab

You can easily try our OoC feature interactively from the Colab link below. Please give it a try!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/out_of_core_training.ipynb)

## Abstract
While large neural networks demonstrate higher performance in various tasks, training large networks is difficult due to limitations on GPU memory size. We propose a novel out-of-core algorithm that enables faster training of extremely large-scale neural networks with sizes larger than allotted GPU memory. Under a given memory budget constraint, our scheduling algorithm locally adapts the timing of memory transfers according to memory usage of each function, which improves overlap between computation and memory transfers. Additionally, we apply virtual addressing technique, commonly performed in OS, to training of neural networks with out-of-core execution, which drastically reduces the amount of memory fragmentation caused by frequent memory transfers. With our proposed algorithm, we successfully train ResNet-50 with 1440 batch-size with keeping training speed at 55%, which is 7.5x larger than the upper bound of physical memory. It also outperforms a previous state-of-the-art substantially, i.e. it trains a 1.55x larger network than state-of-the-art with faster execution. Moreover, we experimentally show that our approach is also scalable for various types of networks.