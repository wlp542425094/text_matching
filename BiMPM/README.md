BiMPM这个模型最大的创新点在于采用了双向多角度匹配，不单单只考虑一个维度，采用了matching-aggregation的结构，把两个句子之间的单元做相似度计算，最后经过全连接层与softamx层得到最终的结果，不可这也成了其缺点，慢。
