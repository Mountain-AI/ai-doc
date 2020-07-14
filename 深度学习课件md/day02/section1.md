# 文件读取流程

## 学习目标

- 目标
  - 说明TensorFlow文件读取的流程
- 应用
  - 无

### 有四种获取数据到TensorFlow程序的方法：

1. **tf.dataAPI：轻松构建复杂的输入管道。（优选方法，在新版本当中）**
2. **QueueRunner：基于队列的输入管道从TensorFlow图形开头的文件中读取数据。**
3. Feeding：运行每一步时，Python代码提供数据。
4. 预加载数据：TensorFlow图中的常量或变量包含所有数据（对于小数据集）。

## 1、文件读取流程

![AnimatedFileQueues](/images/AnimatedFileQueues.gif)

* 第一阶段将生成文件名来读取它们并将它们排入**文件名队列**。
* 第二阶段对于文件名的队列，进行出队列实例，并且实行内容的解码
* 第三阶段重新入新的队列，这将是新的**样本队列**。

 注：这些操作需要**启动运行这些排队操作的线程**，以便我们的训练循环可以将队列中的内容入队出队操作。

### 1.1 第一阶段

我们称之为构造文件队列，将需要读取的文件装入到一个固定的队列当中

* **tf.train.string_input_producer(string_tensor,shuffle=True)**
  * string_tensor：含有文件名+路径的1阶张量
  * num_epochs:过几遍数据，默认无限过数据
  * return 文件队列

### 1.2、第二阶段

这里需要从队列当中读取文件内容，并且进行解码操作。关于读取内容会有一定的规则

#### 1.2.1 读取文件内容

TensorFlow默认每次只读取一个样本，具体到文本文件读取一行、二进制文件读取指定字节数(最好一个样本)、图片文件默认读取一张图片、TFRecords默认读取一个example

- **tf.TextLineReader:**
  - **阅读文本文件逗号分隔值（CSV）格式,默认按行读取**
  - **return：读取器实例**
- **tf.WholeFileReader:用于读取图片文件**
- **tf.TFRecordReader:**
  - **读取TFRecords文件**
- **tf.FixedLengthRecordReader:二进制文件**
  - **要读取每个记录是固定数量字节的二进制文件**
  - **record_bytes:整型，指定每次读取(一个样本)的字节数**
  - **return：读取器实例**

> 1、他们有共同的读取方法：read(file_queue)：从队列中指定数量内容返回一个Tensors元组（key文件名字，value默认的内容(一个样本)）
>
> 2、由于默认只会读取一个样本，所以通常想要进行批处理。使用tf.train.batch或tf.train.shuffle_batch进行多样本获取，便于训练时候指定每批次多个样本的训练

#### 1.2.2 内容解码

对于读取不通的文件类型，内容需要解码操作，解码成统一的Tensor格式

- **tf.decode_csv：解码文本文件内容**
- **tf.decode_raw：解码二进制文件内容**
  - **与tf.FixedLengthRecordReader搭配使用，二进制读取为uint8格式**
- **tf.image.decode_jpeg(contents)**
  - **将JPEG编码的图像解码为uint8张量**
  - **return:uint8张量，3-D形状[height, width, channels]**
- **tf.image.decode_png(contents)**
  - **将PNG编码的图像解码为uint8张量**
  - **return:张量类型，3-D形状[height, width, channels]**

> 解码阶段，默认所有的内容都解码成tf.uint8格式，如果需要后续的类型处理继续处理

### 1.3 第三阶段

在解码之后，我们可以直接获取默认的一个样本内容了，但是如果想要获取多个样本，这个时候需要结合管道的末尾进行批处理

- tf.train.batch(tensors,batch_size,num_threads = 1,capacity = 32,name=None)
  - 读取指定大小（个数）的张量
  - tensors：可以是包含张量的列表,批处理的内容放到列表当中
  - batch_size:从队列中读取的批处理大小
  - num_threads：进入队列的线程数
  - capacity：整数，队列中元素的最大数量
  - return:tensors
- tf.train.shuffle_batch

## 2、线程操作

以上的创建这些队列和排队操作称之为tf.train.QueueRunner。每个QueueRunner都负责一个阶段，并拥有需要在线程中运行的排队操作列表。一旦图形被构建， tf.train.start_queue_runners 函数就会要求图中的每个QueueRunner启动它的运行排队操作的线程。（这些操作需要在会话中开启）

* tf.train.start_queue_runners(sess=None,coord=None)
  * 收集所有图中的队列线程，并启动线程
  * sess:所在的会话中
  * coord：线程协调器
  * return：返回所有线程
* tf.train.Coordinator()
  * 线程协调员,实现一个简单的机制来协调一组线程的终止
  * request_stop()：请求停止
  * should_stop()：询问是否结束
  * join(threads=None, stop_grace_period_secs=120)：回收线程
  * return:线程协调员实例



