ABCNN 论文地址：https://arxiv.org/pdf/1512.05193.pdf

参考博文：https://blog.csdn.net/u012526436/article/details/90179481

作者采用了CNN的结构来提取特征，并用attention机制进行进一步的特征处理，作者一共提出了三种attention的建模方法。

BCNN包括四部分，分别是Input Layer、Convolution Layer、Average Pooling Layer、Output Layer，第一层为Input Layer，第二层与第三层为卷积与池化层，最后是用LR作为Output Layer

ABCNN作者提出了三种结构：ABCNN-1、ABCNN-2、ABCNN-1 + ABCNN-2

ABCNN1与ABCNN2有三个主要的区别。

1是对卷积之前的值进行处理，2是对池化之前的值进行处理
1需要两个权重矩阵W0、W1,参数多一些，更容易过拟合，2不存在这个问题
由于池化是在卷积之后执行的，因此其处理的粒度单元比卷积大，在卷积阶段，获取到的是词/字向量，而池化层获取到的是短语向量，而这个短语向量的维度主要看卷积层的卷积核大小，因此，1和2其实是在不同的粒度上进行了处理。
