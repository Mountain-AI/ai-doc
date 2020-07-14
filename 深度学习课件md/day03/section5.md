# 实战：验证码图片识别

## 学习目标

- 目标
  - 说明验证码识别的原理
  - 说明全连接层的输出设置
  - 说明输出结果的损失、准确率计算
  - 说明验证码标签值的数字转换
  - 应用tf.one_hot实现验证码目标值的one_hot编码处理
- 应用
  - 应用神经网络识别验证码图片

## 1、识别效果

![验证码训练效果](/images/验证码训练效果.png)

## 2、验证码识别实战

![验证码程序结构](/images/验证码程序结构.png)

- 处理原始数据
  - 方便特征值、目标值读取训练
- 设计网络结构
  - 网络的输出处理
- 训练模型并预测

### 原理分析

![分割整体识别](/images/%E5%88%86%E5%89%B2%E6%95%B4%E4%BD%93%E8%AF%86%E5%88%AB.png)

- 1、目标标签分析

![验证码标签分析](/images/%E9%AA%8C%E8%AF%81%E7%A0%81%E6%A0%87%E7%AD%BE%E5%88%86%E6%9E%90.png)

考虑每个位置的可能性？"ABCDEFGHIJKLMNOPQRSTUVWXYZ"

第一个位置：26种可能性

第二个位置：26种可能性

第三个位置：26种可能性

第四个位置：26种可能性

如何比较输出结果和真实值的正确性？可以对每个位置进行one_hot编码

- 2、网络输出分析

按照这样的顺序，"ABCDEFGHIJKLMNOPQRSTUVWXYZ"

```python
真实值：
第一个位置：[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
第二个位置：[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
第三个位置：[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
第四个位置：[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
```

那么每个验证码的目标有[4, 26]这样一个数组

- 3、如何衡量损失

我们考虑将目标值拼接在一起，形成一个[104]长度的一阶张量

```python
真实值：
[0,0,0,0,...0,0,1,0,0][0,0,0,1,...0,0,0,0,0][0,0,0,0,...0,0,0,1,0][1,0,0,0,...0,0,0,0,0]
		  26                    26                   26                     26

预测概率值：
[0.001,0.01,,...,0.2,][0.001,0.01,,...,0.2,][0.001,0.01,,...,0.2,][0.02,0.01,,...,0.1,]
		  26                    26                   26                     26
```

这两个104的一阶张量进行交叉熵损失计算，得出损失大小。**会提高四个位置的概率，使得4组中每组26个目标值中为1的位置对应的预测概率值越来越大，在预测的四组当中概率值最大。这样得出预测中每组的字母位置。**

- 所有104个概率相加为1



- 4、准确率如何计算

预测值和目标值形状要变为[None, 4, 26]，即可这样去比较

![验证码准确率计算](/images/%E9%AA%8C%E8%AF%81%E7%A0%81%E5%87%86%E7%A1%AE%E7%8E%87%E8%AE%A1%E7%AE%97.png)

**在每个验证码的第三个维度去进行比较，4个标签的目标值位置与预测概率位置是否相等，4个全相等，这个样本才预测正确**

```python
维度位置比较：
    0   1   2
[None, 4, 26]

tf.argmax(y_predict, 2)
```

### 3.1 处理原始图片标签数据到TFRecords

#### 3.1.1 验证码原始数据

![验证码图片](/images/验证码图片.png)

![验证码标签数据](/images/验证码标签数据.png)

#### 3.1.2 处理分析

* 处理特征值

避免读取的时候文件名字混乱，自己构造的0~5999的验证码图片文件名字列表

```python
def get_captcha_image():
    """
    获取验证码图片数据
    :param file_list: 路径+文件名列表
    :return: image
    """
    # 构造文件名
    filename = []

    for i in range(6000):
        string = str(i) + ".jpg"
        filename.append(string)

    # 构造路径+文件
    file_list = [os.path.join(FLAGS.captcha_dir, file) for file in filename]

    # 构造文件队列
    file_queue = tf.train.string_input_producer(file_list, shuffle=False)

    # 构造阅读器
    reader = tf.WholeFileReader()

    # 读取图片数据内容
    key, value = reader.read(file_queue)

    # 解码图片数据
    image = tf.image.decode_jpeg(value)

    image.set_shape([20, 80, 3])

    # 批处理数据 [6000, 20, 80, 3]
    image_batch = tf.train.batch([image], batch_size=6000, num_threads=1, capacity=6000)

    return image_batch
```

* 目标值处理

目标值怎么处理，我们每个图片的目标值都是一个字符串。那么将其当做一个个的字符单独处理。一张验证码的图片的目标值由4个数字组成。建立这样的对应关系

```python
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
0,1,2,................,24,25

最终：
"NZPP"----> [[13, 25, 15, 15]]
```

然后将所有的目标值都变成四个数字，然后与对应的特征值一起存入example当中

```python
[[13, 25, 15, 15], [22, 10, 7, 10], [22, 15, 18, 9], [16, 6, 13, 10], [1, 0, 8, 17], [0, 9, 24, 14].....]
```

代码部分：

读取label文件

```python
def get_captcha_label():
    """
    读取验证码图片标签数据
    :return: label
    """
    file_queue = tf.train.string_input_producer(["../data/Genpics/labels.csv"], shuffle=False)

    reader = tf.TextLineReader()

    key, value = reader.read(file_queue)

    records = [[1], ["None"]]

    number, label = tf.decode_csv(value, record_defaults=records)

    # [["NZPP"], ["WKHK"], ["ASDY"]]
    label_batch = tf.train.batch([label], batch_size=6000, num_threads=1, capacity=6000)

    return label_batch
```

处理目标值

```python
# [b'NZPP' b'WKHK' b'WPSJ' ..., b'FVQJ' b'BQYA' b'BCHR']
label_str = sess.run(label)

print(label_str)

# 处理字符串标签到数字张量
label_batch = dealwithlabel(label_str)
```

转换对应的数字

```python
def dealwithlabel(label_str):

    # 构建字符索引 {0：'A', 1:'B'......}
    num_letter = dict(enumerate(list(FLAGS.letter)))

    # 键值对反转 {'A':0, 'B':1......}
    letter_num = dict(zip(num_letter.values(), num_letter.keys()))

    print(letter_num)

    # 构建标签的列表
    array = []

    # 给标签数据进行处理[[b"NZPP"]......]
    for string in label_str:

        letter_list = []# [1,2,3,4]

        # 修改编码，b'FVQJ'到字符串，并且循环找到每张验证码的字符对应的数字标记
        for letter in string.decode('utf-8'):
            letter_list.append(letter_num[letter])

        array.append(letter_list)

    # [[13, 25, 15, 15], [22, 10, 7, 10], [22, 15, 18, 9], [16, 6, 13, 10], [1, 0, 8, 17], [0, 9, 24, 14].....]
    print(array)

    # 将array转换成tensor类型
    label = tf.constant(array)
```

* 特征值、目标值一一对应构造example并写入文件

同一个图片的特征值目标值由于都是非0维数组，所以都以bytes存入

```python
def write_to_tfrecords(image_batch, label_batch):
    """
    将图片内容和标签写入到tfrecords文件当中
    :param image_batch: 特征值
    :param label_batch: 标签纸
    :return: None
    """
    # 转换类型
    label_batch = tf.cast(label_batch, tf.uint8)

    print(label_batch)

    # 建立TFRecords 存储器
    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecords_dir)

    # 循环将每一个图片上的数据构造example协议块，序列化后写入
    for i in range(6000):
        # 取出第i个图片数据，转换相应类型,图片的特征值要转换成字符串形式
        image_string = image_batch[i].eval().tostring()

        # 标签值，转换成整型
        label_string = label_batch[i].eval().tostring()

        # 构造协议块
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))
        }))

        writer.write(example.SerializeToString())

    # 关闭文件
    writer.close()

    return None
```

### 3.2 读取数据训练 

#### 3.2.1 读取TFRecords文件数据

```python
def read_captcha_tfrecords():
    """
    从tfrecords读取图片特征值和目标值
    :return: 特征值、目标值
    """
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer([FLAGS.captcha_tfrecords])

    # 2、构造读取器去读取数据，默认一个样本
    reader = tf.TFRecordReader()

    key, values = reader.read(file_queue)

    # 3、解析example协议
    feature = tf.parse_single_example(values, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string),
    })

    # 4、对bytes类型的数据进行解码
    image = tf.decode_raw(feature['image'], tf.uint8)

    label = tf.decode_raw(feature['label'], tf.uint8)

    print(image, label)

    # 固定每一个数据张量的形状
    image_reshape = tf.reshape(image, [FLAGS.height, FLAGS.width, FLAGS.channel])

    label_reshape = tf.reshape(label, [FLAGS.label_num])

    print(image_reshape, label_reshape)

    # 处理数据的类型
    # 对特征值进行类型修改
    image_reshape = tf.cast(image_reshape, tf.float32)

    label_reshape = tf.cast(label_reshape, tf.int32)

    # 5、进行批处理
    # 意味着每批次训练的样本数量
    image_batch, label_batch = tf.train.batch([image_reshape, label_reshape], batch_size=100, num_threads=1, capacity=100)

    print(image_batch, label_batch)

    return image_batch, label_batch
```

#### 3.2.2 标签数据处理成三维

```python
def change_to_onehot(label_batch):
    """
    处理图片的四个目标值到ont_hot编码
    :param label_batch: [[13, 25, 15, 15], [22, 10, 7, 10], [22, 15, 18, 9]]
    :return: ont_hot
    """

    # [100, 4]---->[100, 4, 26]
    y_true = tf.one_hot(label_batch, depth=FLAGS.depth, on_value=1.0)

    return y_true
```

####3.2.3 全连接层模型建立

每个样本的目标值4个，每个目标值26中可能性，全连接层神经元个数4*26个

```python
def captcha_model(image_batch):
    """
    定义验证码的神经网络模型，得出模型输出
    :param image_batch: 模型的输入数据
    :return: 模型输出结果(预测结果)
    """

    # 直接使用一层  全连接层的神经网络进行预测
    # 确定全连接层的模型计算
    # 输入：[100, 20, 80, 3]         输出：[None, 104]   104 = 4个目标值 * 26中可能性
    with tf.variable_scope("captcha_model"):

        # [100, 20 * 80 * 3]*[20*80*3, 104]+[104] = [None, 104]
        # 随机初始化全连接层的权重和偏置
        w = weight_variables([20 * 80 * 3, 104])

        b = bias_variables([104])

        # 做出全连接层的形状改变[100, 20, 80, 3] ----->[100, 20 * 80 * 3]
        image_reshape = tf.reshape(image_batch, [-1, FLAGS.height * FLAGS.width * FLAGS.channel])

        # 进行矩阵运算
        # y_predict   [None, 104]
        y_predict = tf.matmul(image_reshape, w) + b

    return y_predict
```

#### 3.2.4 计算交叉熵损失

每个图片的104个预测概率与104个真实值之间进行交叉熵计算

```python
# 3、softmax运算计算交叉熵损失
with tf.variable_scope("softmax_crossentropy"):
    # y_true:真实值 [100, 4, 26]  one_hot---->[100, 4 * 26]
    # y_predict :全脸层的输出[100, 104]
    # 返回每个样本的损失组成的列表
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y_true, [100, FLAGS.label_num * FLAGS.depth]),
                                                                      logits=y_predict)
```

#### 3.2.5 得出准确率

形状：[100, 4, 26]的低三个维度进行比较最大值位置

```python
# 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
with tf.variable_scope("accuracy"):
    # 准确率计算需要三维数据对比
    # y_true:真实值 [100, 4, 26]
    # y_predict :全脸层的输出[100, 104]--->[100, 4, 26]
    equal_list = tf.equal(
    tf.argmax(y_true, 2),
    tf.argmax(tf.reshape(y_predict, [100, FLAGS.label_num, FLAGS.depth]), 2)
    )

    accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(equal_list, 1), tf.float32))
```

需要用到一个函数处理equal_list

```python
​```python
    x = tf.constant([[True,  True], [False, False]])
    tf.reduce_all(x)     # False
    tf.reduce_all(x, 0)  # [False, False]
    tf.reduce_all(x, 1)  # [True, False]
​```
```

#### 3.2.6 封装连个参数工具函数

```python
# 封装两个初始化参数的API，以变量Op定义
def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


def bias_variables(shape):
    b = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return b
```

### 3.3 模型训练

```python
def captcha_reco():
    """
    四个目标值的验证码图片识别
    :return:
    """
    # 1、从tfrecords读取图片特征值和目标值
    # image_batch [100, 20, 80, 3]
    # label_batch [100, 4]  [[13, 25, 15, 15], [22, 10, 7, 10], [22, 15, 18, 9]]
    image_batch, label_batch = read_captcha_tfrecords()

    # 2、建立识别验证码的神经网络模型
    # y_predict-->[100, 104]
    y_predict = captcha_model(image_batch)

    # 对目标值进行one_hot编码处理
    # y_true是一个三维形状[100, 4, 26]
    y_true = change_to_onehot(label_batch)

    # 3、softmax运算计算交叉熵损失
    with tf.variable_scope("softmax_crossentropy"):
        # y_true:真实值 [100, 4, 26]  one_hot---->[100, 4 * 26]
        # y_predict :全脸层的输出[100, 104]
        # 返回每个样本的损失组成的列表
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y_true, [100, FLAGS.label_num * FLAGS.depth]),
                                                                      logits=y_predict)
                              )
    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
    with tf.variable_scope("accuracy"):
        # 准确率计算需要三维数据对比
        # y_true:真实值 [100, 4, 26]
        # y_predict :全脸层的输出[100, 104]--->[100, 4, 26]
        equal_list = tf.equal(
            tf.argmax(y_true, 2),
            tf.argmax(tf.reshape(y_predict, [100, FLAGS.label_num, FLAGS.depth]), 2)
        )

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量的op
    init_op = tf.global_variables_initializer()

    # 开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)

        # 创建线程去开启读取任务
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess, coord=coord)

        # sess.run([image_batch, label_batch])
        # 循环训练
        for i in range(1000):

            sess.run(train_op)

            print("第%d步的验证码训练准确率为：%f" % (i,
                                         accuracy.eval()
                                         ))

        # 回收线程
        coord.request_stop()

        coord.join(threads)

    return None
```

### 3.3 保存模型预测

```python
if i % 100 == 0:

	saver.save(sess, "./tmp/model/captcha_model")
```



## 完整训练代码

```python
import tensorflow as tf


class CaptchaIdentification(object):
    """
    验证码识别
    """
    def __init__(self):
        # 验证码属性
        self.height = 20
        self.width = 80
        self.channel = 3

        self.label_num = 4
        self.label_prob = 26

    @staticmethod
    def weight_variables(shape):
        w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
        return w

    @staticmethod
    def bias_variables(shape):
        b = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
        return b

    def read_captcha_tfrecords(self, captcha_tfrecords):
        """
        从tfrecords读取图片特征值和目标值
        :return: 特征值、目标值
        """
        # 1、构造文件队列
        file_queue = tf.train.string_input_producer([captcha_tfrecords])

        # 2、构造读取器去读取数据，默认一个样本
        reader = tf.TFRecordReader()

        key, values = reader.read(file_queue)

        # 3、解析example协议
        feature = tf.parse_single_example(values, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.string),
        })

        # 4、对bytes类型的数据进行解码
        image = tf.decode_raw(feature['image'], tf.uint8)

        label = tf.decode_raw(feature['label'], tf.uint8)

        print(image, label)

        # 固定每一个数据张量的形状
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        label_reshape = tf.reshape(label, [self.label_num])

        print(image_reshape, label_reshape)

        # 处理数据的类型
        # 对特征值进行类型修改
        image_reshape = tf.cast(image_reshape, tf.float32)

        label_reshape = tf.cast(label_reshape, tf.int32)

        # 5、进行批处理
        # 意味着每批次训练的样本数量
        image_batch, label_batch = tf.train.batch([image_reshape, label_reshape], batch_size=100, num_threads=1,
                                                  capacity=100)

        print(image_batch, label_batch)

        return image_batch, label_batch

    def captcha_nn_model(self, image_batch):
        """
        定义验证码的神经网络模型，得出模型输出
        :param image_batch: 模型的输入数据
        :return: 模型输出结果(预测结果)
        """
        # 直接使用一层  全连接层的神经网络进行预测
        # 确定全连接层的模型计算
        # 输入：[100, 20, 80, 3]         输出：[None, 104]   104 = 4个目标值 * 26中可能性
        with tf.variable_scope("captcha_model"):
            # [100, 20 * 80 * 3]*[20*80*3, 104]+[104] = [None, 104]
            # 随机初始化全连接层的权重和偏置
            w = self.weight_variables([20 * 80 * 3, 104])

            b = self.bias_variables([104])

            # 做出全连接层的形状改变[100, 20, 80, 3] ----->[100, 20 * 80 * 3]
            image_reshape = tf.reshape(image_batch, [-1, self.height * self.width * self.channel])

            # 进行矩阵运算
            # y_predict   [None, 104]
            y_predict = tf.matmul(image_reshape, w) + b

        return y_predict

    def change_to_onehot(self, label_batch):
        """
        处理图片的四个目标值到ont_hot编码
        :param label_batch: [[13, 25, 15, 15], [22, 10, 7, 10], [22, 15, 18, 9]]
        :return: ont_hot
        """

        # [100, 4]---->[100, 4, 26]
        y_true = tf.one_hot(label_batch, depth=self.label_prob, on_value=1.0)

        return y_true

    def loss(self, y_true, y_predict):
        """
        计算损失
        :return:
        """
        # 3、softmax运算计算交叉熵损失
        with tf.variable_scope("softmax_crossentropy"):
            # y_true:真实值 [100, 4, 26]  one_hot---->[100, 4 * 26]
            # y_predict :全脸层的输出[100, 104]
            # 返回每个样本的损失组成的列表
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.reshape(y_true, [100, self.label_num * self.label_prob]),
                    logits=y_predict)
            )

        return loss

    def sgd(self, loss):
        """
        梯度下降优化
        :param loss:
        :return:
        """
        # 4、梯度下降损失优化
        with tf.variable_scope("optimizer"):
            # 学习率
            train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        return train_op

    def accuracy(self, y_true, y_predict):
        """
        计算准确率
        :param y_true:
        :param y_predict:
        :return:
        """
        # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
        with tf.variable_scope("accuracy"):
            # 准确率计算需要三维数据对比
            # y_true:真实值 [100, 4, 26]
            # y_predict :全脸层的输出[100, 104]--->[100, 4, 26]
            equal_list = tf.equal(
                tf.argmax(y_true, 2),
                tf.argmax(tf.reshape(y_predict, [100, self.label_num, self.label_prob]), 2)
            )

            accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(equal_list, 1), tf.float32))

        return accuracy

    def train(self):
        """
        四个目标值的验证码图片识别
        :return:
        """
        # 1、从tfrecords读取图片特征值和目标值
        # image_batch [100, 20, 80, 3]
        # label_batch [100, 4]  [[13, 25, 15, 15], [22, 10, 7, 10], [22, 15, 18, 9]]
        image_batch, label_batch = self.read_captcha_tfrecords("../code/CaptchaRecognize/tfrecords/captcha.tfrecords")

        # 2、建立识别验证码的神经网络模型
        # y_predict-->[100, 104]
        y_predict = self.captcha_nn_model(image_batch)

        # 对目标值进行one_hot编码处理
        # y_true是一个三维形状[100, 4, 26]
        y_true = self.change_to_onehot(label_batch)

        loss = self.loss(y_true, y_predict)

        train_op = self.sgd(loss)

        accuracy = self.accuracy(y_true, y_predict)

        saver = tf.train.Saver()

        # 开启会话运行
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 创建线程去开启读取任务
            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(sess, coord=coord)

            # sess.run([image_batch, label_batch])
            # 循环训练
            for i in range(1000):
                sess.run(train_op)

                print("第%d步的验证码训练准确率为：%f" % (i,
                                             accuracy.eval()
                                             ))
                if i % 100 == 0:

                    saver.save(sess, "./model/captcha_model")

            # 回收线程
            coord.request_stop()

            coord.join(threads)

        return None


def main(argv):

    CI = CaptchaIdentification()

    CI.train()


if __name__ == '__main__':
    tf.app.run()
```

## 4、拓展

* 如果验证码的标签值不止是大写字母，比如还包含小写字母和数字，该怎么处理？
* 如果图片的目标值不止4个，可能5，6个，该怎么处理？

注：主要是在网络输出的结果以及数据对应数字进行分析