ESIM模型主要是用来做文本推理的，给定一个前提premise p 推导出假设hypothesis h，其损失函数的目标是判断p与h是否有关联，即是否可以由p推导出h
该模型也可以做文本匹配，只是损失函数的目标是两个序列是否是同义句。接下来我们就从模型的结构开始讲解其原理。
参考博客：https://blog.csdn.net/u012526436/article/details/90380840
参考代码：https://github.com/terrifyzhao/text_matching

模型网络：graph.py 
模型配置：args.py
加载数据：utils.py
运行文件：train.py
