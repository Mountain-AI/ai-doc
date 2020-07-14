# 深度学习框架介绍

## 学习目标

- 目标
  - 了解常见的深度学习框架
  - 了解TensorFlow框架

- 应用
  - 无

## 1、常见深度学习框架对比

![框架关注](/images/框架关注.png)

| **比较项**   | **Caffe**     | **Torch**    | **Theano**      | **TensorFlow** | **MXNet**         |
| ------------ | ------------- | ------------ | --------------- | -------------- | ----------------- |
| **主语言**   | C++/cuda      | C++/Lua/cuda | Python/c++/cuda | C++/cuda       | C++/cuda          |
| **从语言**   | Python/Matlab | -            | -               | Python         | Python/R/Julia/Go |
| **硬件**     | CPU/GPU       | CPU/GPU/FPGA | CPU/GPU         | CPU/GPU/Mobile | CPU/GPU/Mobile    |
| **分布式**   | N             | N            | N               | Y(未开源)      | Y                 |
| **速度**     | 快            | 快           | 中等            | 中等           | 快                |
| **文档**     | 全面          | 全面         | 中等            | 中等           | 全面              |
| **适合模型** | CNN           | CNN/RNN      | CNN/RNN         | CNN/RNN        | CNN               |
| **操作系统** | 所有系统      | Linux, OSX   | 所有系统        | Linux, OSX     | 所有系统          |
| **命令式**   | N             | Y            | N               | N              | Y                 |
| **声明式**   | Y             | N            | Y               | Y              | Y                 |
| **接口**     | protobuf      | Lua          | Python          | C++/Python     | Python/R/Julia/Go |
| **网络结构** | 分层方法      | 分层方法     | 符号张量图      | 符号张量图     | 无                |

## 2、TensorFlow的特点

![tensorflow](/images/tensorflow.png)

官网：https://www.tensorflow.org/

* 高度灵活（Deep Flexibility）
  * 它不仅是可以用来做神经网络算法研究，也可以用来做普通的机器学习算法，甚至是只要你能够把计算表示成数据流图，都可以用TensorFlow。
* 语言多样（Language Options）
  * TensorFlow使用C++实现的，然后用Python封装。谷歌号召社区通过SWIG开发更多的语言接口来支持TensorFlow
* 设备支持
  * TensorFlow可以运行在各种硬件上，同时根据计算的需要，合理将运算分配到相应的设备，比如卷积就分配到GPU上，也允许在 CPU 和 GPU 上的计算分布，甚至支持使用 gRPC 进行水平扩展。
* Tensorboard可视化
  * TensorBoard是TensorFlow的一组Web应用，用来监控TensorFlow运行过程，或可视化Computation Graph。TensorBoard目前支持5种可视化：标量（scalars）、图片（images）、音频（audio）、直方图（histograms）和计算图（Computation Graph）。TensorBoard的Events Dashboard可以用来持续地监控运行时的关键指标，比如loss、学习速率（learning rate）或是验证集上的准确率（accuracy）

## 3、 TensorFlow的安装

### 3.1 CPU版本

安装较慢，指定镜像源，请在带有numpy等库的虚拟环境中安装

* ubuntu安装

```
pip install tensorflow==1.8 -i https://mirrors.aliyun.com/pypi/simple
```

* MacOS安装

```
pip install tensorflow==1.8 -i https://mirrors.aliyun.com/pypi/simple
```

### 3.2 GPU版本

参考官网：

- [在 Ubuntu 上安装 TensorFlow](https://www.tensorflow.org/install/install_linux)
- [在 macOS 上安装 TensorFlow](https://www.tensorflow.org/install/install_mac)