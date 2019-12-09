ABCNN 论文地址：https://arxiv.org/pdf/1512.05193.pdf

作者采用了CNN的结构来提取特征，并用attention机制进行进一步的特征处理，作者一共提出了三种attention的建模方法。

BCNN包括四部分，分别是Input Layer、Convolution Layer、Average Pooling Layer、Output Layer，第一层为Input Layer，第二层与第三层为卷积与池化层，最后是用LR作为Output Layer

ABCNN作者提出了三种结构：ABCNN-1、ABCNN-2、ABCNN-1 + ABCNN-2
