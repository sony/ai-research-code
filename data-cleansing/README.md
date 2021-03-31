# Data Cleansing with Storage-efficient Approximation of Influence Functions

This is the code for data cleansing : Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions.

> [**Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions**]((https://arxiv.org/abs/2103.11807).),
> Kenji Suzuki, Yoshiyuki Kobayashi, Takuya Narihira
> *arXiv technical report ([arXiv 2103.11807]( https://arxiv.org/abs/2103.11807))*            

![](./imgs/datacleansing.png)

## Code
The code to data cleansing is provided by **resonsible_ai** folder in the sony/nnabla-examples repository.  Please give it a try.

[code](https://github.com/sony/nnabla-examples)

## Abstract 
Identifying the influence of training data for data cleansing can improve the accuracy of deep learning. An approach with stochastic gradient descent (SGD) called SGD-influence to calculate the influence scores was proposed [1], but, the calculation costs are expensive. It is necessary to temporally store the parameters of the model during training phase for inference phase to calculate influence sores. In close connection with the previous method, we propose a method to reduce cache files to store the parameters in training phase for calculating inference score. We only adopt the final parameters in last epoch for influence functions calculation. In our experiments on classification, the cache size of training using MNIST dataset with our approach is 1.236 MB. On the other hand, the previous method used cache size of 1.932 GB in last epoch. It means that cache size has been reduced to 1/1,563. We also observed the accuracy improvement by data cleansing with removal of negatively influential data using our approach as well as the previous method. Moreover, our *simple* and *general* proposed method to calculate influence scores is available on our auto ML tool without programing, Neural Network Console. The source code is also available.

## Citation
@misc{suzuki2021data,
      title={Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions}, 
      author={Kenji Suzuki and Yoshiyuki Kobayashi and Takuya Narihira},
      year={2021},
      eprint={2103.11807},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

## Reference
[1] Satoshi Hara, Atsushi Nitanda, and Takanori Maehara. Data cleansing for models trained with SGD. In *Advances in  Neural Information Processing Systems*, pages 4215-4224, 2019


