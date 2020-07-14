# 图片数据

## 学习目标

- 目标
  - 说明图片数字化的三要素
  - 说明图片三要素与张量的表示关系
  - 了解张量的存储和计算类型
  - 应用tf.image.resize_images实现图像的像素改变
  - 应用tf.train.start_queue_runners实现读取线程开启
  - 应用tf.train.Coordinator实现线程协调器开启
  - 应用tf.train.batch实现数据的批处理
- 应用
  - 狗图片读取

## 1、 图像基本知识

对于图像文件，我们怎么进行转换成机器学习能够理解的数据。之前我们讲过文本怎么处理成数字信息。对于图片来讲，组成图片的最基本单位是像素，所以我们获取的是每张图片的像素值。接触的图片有两种，一种是黑白图片，另一种是彩色图片。

![黑白彩色](/images/黑白彩色.png)

### 1.1 图片三要素

组成一张图片特征值是所有的像素值，有这么几个要素。**图片长度、图片宽度、图片通道数**。什么是图片的通道数呢，描述一个像素点，如果是灰度，那么只需要一个数值来描述它，就是单通道。如果一个像素点，有RGB三种颜色来描述它，就是三通道。那所以

* 灰度图：单通道
* 彩色图片：三通道

![通道数](/images/通道数.png)

> 假设一张彩色图片的长200，宽200，通道数为3，那么总的像素数量为200 * 200 * 3

### 1.2 张量形状

读取图片之后，怎么用张量形状来表示呢。一张图片就是一个3D张量，[height, width, channel]，height就表示高，width表示宽，channel表示通道数。我们会经常遇到3D和4D的表示

* 单个图片：[height, width, channel]
* 多个图片：[batch,height, width, channel]，batch表示批数量

### 1.3 图片特征值处理

在进行图片识别的时候，每个图片样本的特征数量要保持相同。所以需要将所有图片张量大小统一转换。另一方面如果图片的像素量太大，也可以通过这种方式适当减少像素的数量，减少训练的计算开销

* **tf.image.resize_images(images, size)**
  * 缩小放大图片
  * images：4-D形状[batch, height, width, channels]或3-D形状的张量[height, width, channels]的图片数据
  * size：1-D int32张量：new_height, new_width，图像的新尺寸
  * 返回4-D格式或者3-D格式图片

### 1.4 数据格式

* 存储：uint8(节约空间)
* 矩阵计算：float32(提高精度)

## 2、案例：狗图片读取

### 2.1 读取流程分析

- 构造图片文件队列
- 读取图片数据并进行解码
- 处理图片数据形状，批处理返回
- 开启会话线程运行

### 2.2 代码

```python
def picread(file_list):
    """
    狗图片读取，转换成数据张量
    :return:
    """
    # 1、构造文件队列
    # 返回文件队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2、构造一个图片读取器，去对垒当中读取数据
    # 返回reader实例，调用read方法读取内容，key, value
    reader = tf.WholeFileReader()

    key, value = reader.read(file_queue)

    print(value)

    # 3、对样本内容进行解码
    image = tf.image.decode_jpeg(value)

    print(image)

    # 处理图片的大小，形状，resize_images图片数据类型变成了float类型
    image_resize = tf.image.resize_images(image, [200, 200])

    print(image_resize)

    # 设置固定形状，这里可以使用静态形状API去修改
    image_resize.set_shape([200, 200, 3])

    # 4、批处理图片数据
    # 每个样本的形状必须全部定义
    image_batch = tf.train.batch([image_resize], batch_size=100, num_threads=1, capacity=100)

    print(image_batch)

    return image_batch
```

* 会话逻辑

```python
if __name__ == "__main__":
    # 生成路径+文件名的列表
    filename = os.listdir("./data/dog/")

    # 路径+名字拼接
    file_list = [os.path.join("./data/dog/", file) for file in filename]

    # 从原始二进制文件读取
    image_batch, label_batch = picread(file_list)

    # 开启会话打印内容
    with tf.Session() as sess:
        # 创建线程协调器
        coord = tf.train.Coordinator()

        # 开启子线程去读取数据
        # 返回子线程实例
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 获取样本数据去训练
        print(sess.run([image_batch, label_batch]))

        # 关闭子线程，回收
        coord.request_stop()

        coord.join(threads)
```

