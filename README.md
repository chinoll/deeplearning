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
生成效果并不好 <br>
![wgan](https://github.com/chinoll/deeplearning/raw/master/imgs/wgan.png) <br>
在添加batchnorm之后，模式崩溃了<br>

## Auxiliary Classifier GAN(ACGAN)
![wgan](https://github.com/chinoll/deeplearning/raw/master/imgs/acgan.png) <br>
使用梯度标准化(仅生成器使用梯度标准化)，在前期生成的质量更好，在epoch增加，loss也增加，并且生成的图片有模式崩溃 <br>
对生成器和判别器都使用梯度标准化，无法生成正常的图片 <br>
### 使用梯度标准化生成的图片
![acgan](https://github.com/chinoll/deeplearning/raw/master/imgs/acgan2.png) <br>
