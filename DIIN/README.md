DIIN的结构主要分为五层，分别是Embedding Layer、Encoding Layer、Interaction Layer、Feature Extraction Layer、Output Layer。

Embedding Layer会把每一个词或者说段落转变为向量表示，和其他模型不同的点在于其不仅仅只采用了字、词向量，还添加句法特征。

Encoding Layer的主要作用是将上一层的特征进行融合并进行encode

Interaction Layer的主要目的是把P与H做一个相似度的计算，提取出其中的相关性，可以采用余弦相似度、欧氏距离等等，这里作者发现对位相乘的效果很好
