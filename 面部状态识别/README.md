# 面 部 状 态 识 别
-----
<br>
  第一个模块是面部状态识别模块：<br>
  我们选用“fer2013”数据库进行预训练，由于图片大小为48*48，不便于采用层次太深的CNN模型；<br>
  因此我们搭建了仅有2～3个卷积/池化层的CNN模型<br>
  为了加速训练，我去除了Droup—out，换用BN（Batch Normalization）来同时解决梯度弥散和过拟合的问题<br>
  BN方法在CNN模型中已经实现，具体原理部分可以查看“莫烦PYTHON”中的讲解。<br>
  LINK：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-13-A-batch-normalization/<br>

  <br><br><br>
  经过训练（本次训练模型不算太复杂，因此用的是3G版的GTX1060显卡，训练速度也很快），在训练集上精度可达99%，但在验证集和测试集上只有54%左右的精度；<br><br>
  而且，无论如何调整网络结构，甚至选用Droup—out方法都不能带来精度的进一步提升，因此，我推测局限于数据本身和网络复杂度，这个精度已经算是可以的了（我在Kaggle上看到了此类比赛，top-1也只有70%左右的精度）<br><br>
  <br>
  因此，综合考虑之后，我决定采用Bagging集成学习的方法集成一系列异构的CNN网络来达到精度的提升<br>
  通过修改网路结构，包括卷积/池化层的数量与Kernel大小、全连接层的结点数目、droup-out中keep_prob比例、激活函数方法等等方案，构造了总共10个小型CNN模型，精度大约都在45%～56%之间<br><br><br><br><br><br>

  【抱歉，由于近期在准备并行计算和计算机视觉的考试，暂时没有做Bagging集成，大约在两周后会做集成】<br><br><br><br><br><br>

<br>
1.loadData.py--------------数据预处理<br><br>
2.CNN_NO.1.py--------------CNN模型<br><br>
3.CNN_NO.1_test.py----------读取模型并测试<br><br>
4.CNN_NO.1_prediction.py--------CNN预测结果<br><br>
5.twoStream_CNN.py-------双流（光流+图像流）CNN【后续可能会使用】<br><br>
6.Saver---------训练好的模型<br><br>
7.logs----------tensorboard保存结点<br><br>
8.DataSet-------预处理后的.tfrecords文件（太大无法上传，可以自己用代码生成）<br><br>
9.fer2013-------面部数据库（太大无法上传，这个百度就有）<br><br>

<br><br><br><br>

tips:<br>
foxmail：  zixuwang1997@foxmail.com<br>
gamil:     zixuwang1997@gmail.com<br>
others:    zixuwang@csu.edu.cn<br>
