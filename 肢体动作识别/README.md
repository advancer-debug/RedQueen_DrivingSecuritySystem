# 肢 体 动 作 识 别
-----
<br>
  第二个模块是肢体动作识别模块：<br>
  我们选用Kaggle比赛的数据库进行预训练，由于图片大小为640*480，因此我们选用了VGGNet模型；<br>
  由于VGGNet层次非常深，我们手中的设备很难训练如此复杂的网络（主要是显存不够），因此我们采用了Fine-tuned技术来加速训练<br>
  所谓Fine-tuned技术，就是加载别人已经训练好的模型（这里用的是ImageNet上与训练好的VGGNet）<br>
  把VGG中前面的卷积层全部冻结，不进行训练（这是由于卷积层是用来提取高阶抽象特征的，图像的组成部分大都差不多，因此可以直接加载）<br>
  第二就是卷积层在做BP算法的时候计算复杂很多，会用到广义傅立叶变换，速度非常慢<br>
  因此冻结卷积层可以加速训练，相当于只训练全连接层<br>
  更多关于Fine-tuned技术：https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html<br>
  <br>
  <br>
  <br>

  核心技术介绍完了就可以开始训练了，数据集来源：https://www.kaggle.com/c/state-farm-distracted-driver-detection  <br>  <br>
  train目录是给我们训练用的目录，里面一共有c0到c9十个分类，对应我们刚刚列出的 0.正常开车 1.右手玩手机 2.右手打电话 等十种行为。在这个训练集中一共包含22,286张图片。  <br>  <br>  <br>  <br>

  训练的时候就开始无限踩坑了，其中最重要的是、、、我们的显卡只有3G显存，每次每个batch只能放3个样本，不然就炸了，而且全连接层结点只能是512*256，不然就跑不了了<br><br>
  没办法只能硬着头皮跑了，结果损失卡在【2.3】死活下不去，测试集精度也只有10%、、、完全没用<br>
  <br>
  找了一周的bug，各种方法都试过了，都没用<br>
  没办法，怀疑到batch size太小，又申请了一块高级的图形显卡，有8G<br>
  修改了batch size大小，把网络改成了700多个隐藏结点，然后损失就下去了？？？？？？？<br>
  我说的轻松，但这一步花了一个月的时间，尝试了各种方法，其中冷暖自知啊<br>
  <br><br><br><br>
  然后在实验室跑了一天，模型验证集精度能到99%，测试集也差不多90%的样子，可以认为训练好了<br>
  <br><br><br>
  然后开始用我们的照片进行微调，这个过程很顺利，没用多久就可以使用了<br>
  <br><br>
  这个模块差不多就做好了<br>

<br><br><br><br><br>
1.load_transfer_Data.py--------------数据预处理1<br><br>
2.loadData.py--------------数据预处理2<br><br>
3.VGGNet_19.py--------------VGG模型<br><br>
4，vgg16.py/vgg19.py---------加载ImageNet上预训练好的模型（这个模型需要自己下载了，太大没法上传）<br><br>
5.VGGNet_19_testSet.py-------测试代码<br><br>
6.opencv.py--------OpenCV驱动摄像头<br><br>
7.VGGNet_19_camera.py-------从摄像头捕捉图像并预测label<br><br>
8.logs----------tensorboard保存结点<br><br>
9.VGGNet_19-------ImageNet预训练好的模型（太大无法上传）<br><br>
10.model-------训练好的模型（太大无法上传）<br><br>

<br><br><br><br>

tips:<br>
foxmail：  zixuwang1997@foxmail.com<br>
gamil:     zixuwang1997@gmail.com<br>
others:    zixuwang@csu.edu.cn<br>
