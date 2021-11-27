# deeplearning
深度学习经典论文实现
## Deep Convolutional GAN(DCGAN)
### 实验结果
生成器的最后一层为Tanh函数生成的效果 <br>
![dcgan](https://github.com/chinoll/deeplearning/raw/master/imgs/dcgan.png) <br>
将生成器的最后一层换成LRelu之后生成的图像质量更差劲<br>
![dcgan_lrelu](https://github.com/chinoll/deeplearning/raw/master/imgs/dcgan_lrelu.png) <br>
将leakyRelu换成Relu之后生成的图像<br>
![dcgan_relu](https://github.com/chinoll/deeplearning/raw/master/imgs/dcgan_relu.png) <br>

## Wasserstein  GAN(WGAN)
生成效果并不好
![wgan](https://github.com/chinoll/deeplearning/raw/master/imgs/wgan.png) <br>
在添加batchnorm之后，模式崩溃了<br>