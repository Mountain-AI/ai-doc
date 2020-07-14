TFRecords

## 学习目标

- 目标
  - 说明Example的结构
  - 应用TFRecordWriter实现TFRecords文件存储器的构造
  - 应用parse_single_example实现解析Example结构
- 应用
  - CIFAR10类图片的数据的TFRecords存储和读取

## 1、什么是TFRecords文件

TFRecords其实是一种二进制文件，虽然它不如其他格式好理解，但是它能更好的利用内存，更方便复制和移动，**并且不需要单独的标签文件**。

TFRecords文件包含了`tf.train.Example` 协议内存块(protocol buffer)(协议内存块包含了字段 `Features`)。可以获取你的数据， 将数据填入到`Example`协议内存块(protocol buffer)，将协议内存块序列化为一个字符串， 并且通过`tf.python_io.TFRecordWriter` 写入到TFRecords文件。

* 文件格式 *.tfrecords 

## 2、Example结构解析

`tf.train.Example` 协议内存块(protocol buffer)(协议内存块包含了字段 `Features`)，`Features`包含了一个`Feature`字段，`Features`中包含要写入的数据、并指明数据类型。这是一个样本的结构，批数据需要循环存入这样的结构

```python
 example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
```

- tf.train.Example(**features**=None)
  - 写入tfrecords文件
  - features:tf.train.Features类型的特征实例
  - return：example格式协议块
- tf.train.**Features**(**feature**=None)
  - 构建每个样本的信息键值对
  - feature:字典数据,key为要保存的名字
  - value为tf.train.Feature实例
  - return:Features类型
- tf.train.**Feature**(options)
  - options：例如
    - bytes_list=tf.train. BytesList(value=[Bytes])
    - int64_list=tf.train. Int64List(value=[Value])
  - 支持存入的类型如下
  - tf.train.Int64List(value=[Value])
  - tf.train.BytesList(value=[Bytes]) 
  - tf.train.FloatList(value=[value]) 

> 这种结构是不是很好的解决了**数据和标签(训练的类别标签)或者其他属性数据存储在同一个文件中** 

```python

```

## 3、案例：CIFAR10数据存入TFRecords文件

### 3.1分析

- 构造存储实例，tf.python_io.TFRecordWriter(path)
  * 写入tfrecords文件
  * path: TFRecords文件的路径
  * return：写文件
  * method
  * write(record):向文件中写入一个example
  * close():关闭文件写入器

* 循环将数据填入到`Example`协议内存块(protocol buffer)

### 3.2代码 

```python
    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将数据存进tfrecords，方便管理每个样本的属性
        :param image_batch: 特征值
        :param label_batch: 目标值
        :return: None
        """
        # 1、构造tfrecords的存储实例
        writer = tf.python_io.TFRecordWriter(FLAGS.tfrecords_dir)

        # 2、循环将每个样本写入到文件当中
        for i in range(10):

            # 一个样本一个样本的处理写入
            # 准备特征值，特征值必须是bytes类型 调用tostring()函数
            # [10, 32, 32, 3] ，在这里避免tensorflow的坑，取出来的不是真正的值，而是类型，所以要运行结果才能存入
            # 出现了eval,那就要在会话当中去运行该行数
            image = image_batch[i].eval().tostring()

            # 准备目标值，目标值是一个Int类型
            # eval()-->[6]--->6
            label = label_batch[i].eval()[0]

            # 绑定每个样本的属性
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))

            # 写入每个样本的example
            writer.write(example.SerializeToString())

        # 文件需要关闭
        writer.close()
        return None
    
    # 开启会话打印内容
    with tf.Session() as sess:
        # 创建线程协调器
        coord = tf.train.Coordinator()

        # 开启子线程去读取数据
        # 返回子线程实例
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 获取样本数据去训练
        print(sess.run([image_batch, label_batch]))

        # 存入数据
        cr.write_to_tfrecords(image_batch, label_batch )

        # 关闭子线程，回收
        coord.request_stop()

        coord.join(threads)
```

## 4、读取TFRecords文件

读取这种文件整个过程与其他文件一样，只不过需要有个解析Example的步骤。从TFRecords文件中读取数据， 可以使用`tf.TFRecordReader`的`tf.parse_single_example`解析器。这个操作可以将`Example`协议内存块(protocol buffer)解析为张量。

```python
# 多了解析example的一个步骤
        feature = tf.parse_single_example(values, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
```

* tf.parse_single_example(serialized,features=None,name=None)
  * 解析一个单一的Example原型
  * serialized：标量字符串Tensor，一个序列化的Example
  * features：dict字典数据，键为读取的名字，值为FixedLenFeature
  * return:一个键值对组成的字典，键为读取的名字

- tf.FixedLenFeature(shape,dtype)
  - shape：输入数据的形状，一般不指定,为空列表
  - dtype：输入数据类型，与存储进文件的类型要一致
  - 类型只能是float32,int64,string

### 5、案例：读取CIFAR的TFRecords文件

### 5.1 分析

-  使用tf.train.string_input_producer构造文件队列
- tf.TFRecordReader 读取TFRecords数据并进行解析
  - tf.parse_single_example进行解析
- tf.decode_raw解码
  - 类型是bytes类型需要解码
  - 其他类型不需要
- **处理图片数据形状以及数据类型**，批处理返回
- 开启会话线程运行

### 5.2 代码

```python
    def read_tfrecords(self):
        """
        读取tfrecords的数据
        :return: None
        """
        # 1、构造文件队列
        file_queue = tf.train.string_input_producer(["./tmp/cifar.tfrecords"])

        # 2、构造tfrecords读取器，读取队列
        reader = tf.TFRecordReader()

        # 默认也是只读取一个样本
        key, values = reader.read(file_queue)

        # tfrecords
        # 多了解析example的一个步骤
        feature = tf.parse_single_example(values, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })

        # 取出feature里面的特征值和目标值
        # 通过键值对获取
        image = feature["image"]

        label = feature["label"]

        # 3、解码操作
        # 对于image是一个bytes类型，所以需要decode_raw去解码成uint8张量
        # 对于Label:本身是一个int类型，不需要去解码
        image = tf.decode_raw(image, tf.uint8)

        print(image, label)

        # # 从原来的[32,32,3]的bytes形式直接变成[32,32,3]
        # 不存在一开始我们的读取RGB的问题
        # 处理image的形状和类型
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        # 处理label的形状和类型
        label_cast = tf.cast(label, tf.int32)

        print(image_reshape, label_cast)

        # 4、批处理操作
        image_batch, label_batch = tf.train.batch([image_reshape, label_cast], batch_size=10, num_threads=1, capacity=10)

        print(image_batch, label_batch)
        return image_batch, label_batch
    
    
   
    
    # 从tfrecords文件读取数据
    image_batch, label_batch = cr.read_tfrecords()

    # 开启会话打印内容
    with tf.Session() as sess:
        # 创建线程协调器
        coord = tf.train.Coordinator()
```

