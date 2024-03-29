# 案例：实现线性回归

## 学习目标

- 目标
  - 应用op的name参数实现op的名字修改
  - 应用variable_scope实现图程序作用域的添加
  - 应用scalar或histogram实现张量值的跟踪显示
  - 应用merge_all实现张量值的合并
  - 应用add_summary实现张量值写入文件
  - 应用tf.train.saver实现TensorFlow的模型保存以及加载
  - 应用tf.app.flags实现命令行参数添加和使用
  - 应用reduce_mean、square实现均方误差计算
  - 应用tf.train.GradientDescentOptimizer实现有梯度下降优化器创建
  - 应用minimize函数优化损失
  - 知道梯度爆炸以及常见解决技巧
- 应用
  - 实现线性回归模型

## 1、线性回归原理复习

根据数据建立回归模型，w1x1+w2x2+…..+b = y，通过真实值与预测值之间建立误差，使用梯度下降优化得到损失最小对应的权重和偏置。最终确定模型的权重和偏置参数。最后可以用这些参数进行预测。

## 2、案例：实现线性回归的训练

### 2.1 案例确定

* 假设随机指定100个点，只有一个特征
* 数据本身的分布为 y = 0.7 * x + 0.8

> 这里将数据分布的规律确定，是为了使我们训练出的参数跟真实的参数（即0.7和0.8）比较是否训练准确

### 2.2 API

**运算**

* 矩阵运算
  * tf.matmul(x, w)
* 平方
  * tf.square(error)
* 均值
  * tf.reduce_mean(error)

**梯度下降优化**

* tf.train.GradientDescentOptimizer(learning_rate)
  * 梯度下降优化
  * learning_rate:学习率，一般为0~1之间比较小的值
  * method:
    * minimize(loss)
  * return:梯度下降op

### 2.3 步骤分析

* 1、准备数据的特征值和目标值  inputs
* 2、根据特征值建立线性回归模型(确定参数个数形状) inference
  * 模型的参数必须使用变量OP创建
* 3、根据模型得出预测结果，建立损失 loss
* 4、梯度下降优化器优化损失 sgd_op

### 2.4 实现完整功能

```python
    def inputs(self):
        """
        获取特征值目标值数据数据
        :return:
        """
        x_data = tf.random_normal([100, 1], mean=1.0, stddev=1.0, name="x_data")
        y_true = tf.matmul(x_data, [[0.7]]) + 0.8

        return x_data, y_true

    def inference(self, feature):
        """
        根据输入数据建立模型
        :param feature:
        :param label:
        :return:
        """
        with tf.variable_scope("linea_model"):
            # 2、建立回归模型，分析别人的数据的特征数量--->权重数量， 偏置b
            # 由于有梯度下降算法优化，所以一开始给随机的参数，权重和偏置
            # 被优化的参数，必须得使用变量op去定义
            # 变量初始化权重和偏置
            # weight 2维[1, 1]    bias [1]
            # 变量op当中会有trainable参数决定是否训练
            self.weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0),
                                 name="weights")

            self.bias = tf.Variable(0.0, name='biases')

            # 建立回归公式去得出预测结果
            y_predict = tf.matmul(feature, self.weight) + self.bias

        return y_predict

    def loss(self, y_true, y_predict):
        """
        目标值和真实值计算损失
        :return: loss
        """
        # 3、求出我们模型跟真实数据之间的损失
        # 均方误差公式
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

        return loss
    
    def sgd_op(self, loss):
        """
        获取训练OP
        :return:
        """
        # 4、使用梯度下降优化器优化
        # 填充学习率：0 ~ 1    学习率是非常小，
        # 学习率大小决定你到达损失一个步数多少
        # 最小化损失
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        return train_op
```

### 2.5 学习率的设置、步长的设置与梯度爆炸

学习率越大，训练到较好结果的步长越小；学习率越小，训练到较好结果的步长越大。

但是学习过大会出现梯度爆炸现象。**关于梯度爆炸/梯度消失 ？**

```python
在极端情况下，权重的值变得非常大，以至于溢出，导致 NaN 值
如何解决梯度爆炸问题（深度神经网络当中更容易出现）
1、重新设计网络
2、调整学习率
3、使用梯度截断（在训练过程中检查和限制梯度的大小）
4、使用激活函数
```

### 2.6 变量的trainable设置观察

trainable的参数作用，指定是否训练

```python
weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="weights", trainable=False)
```

## 3、增加其他功能

* **增加命名空间**
* **变量Tensorboard显示**
* **模型保存与加载**
* **命令行参数设置**

### 3.1 增加命名空间 

是代码结构更加清晰，Tensorboard图结构清楚

```python
with tf.variable_scope("lr_model"):
```

### 3.2 增加变量显示

**目的：在TensorBoard当中观察模型的参数、损失值等变量值的变化** 

* 1、收集变量
  * tf.summary.scalar(name=’’,tensor) 收集对于损失函数和准确率等单值变量,name为变量的名字，tensor为值
  * tf.summary.histogram(name=‘’,tensor) 收集高维度的变量参数
  * tf.summary.image(name=‘’,tensor) 收集输入的图片张量能显示图片
* 2、合并变量写入事件文件
  * merged = tf.summary.merge_all()
  * 运行合并：summary = sess.run(merged)，每次迭代都需运行
  * 添加：FileWriter.add_summary(summary,i),i表示第几次的值

```python
    def merge_summary(self, loss):

        # 1、收集张量的值
        tf.summary.scalar("losses", loss)

        tf.summary.histogram("w", self.weight)
        tf.summary.histogram('b', self.bias)

        # 2、合并变量
        merged = tf.summary.merge_all()

        return merged


# 生成事件文件，观察图结构
file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)


# 运行收集变量的结果
summary = sess.run(merged)

# 添加到文件
file_writer.add_summary(summary, i)
```

### 3.3 模型的保存与加载 

* **tf.train.Saver(var_list=None,max_to_keep=5)**
  * 保存和加载模型（保存文件格式：checkpoint文件）
  * var_list:指定将要保存和还原的变量。它可以作为一个dict或一个列表传递.
  * max_to_keep：指示要保留的最近检查点文件的最大数量。创建新文件时，会删除较旧的文件。如果无或0，则保留所有检查点文件。默认为5（即保留最新的5个检查点文件。）

使用

```python
例如：
指定目录+模型名字
saver.save(sess, '/tmp/ckpt/test/myregression.ckpt')
saver.restore(sess, '/tmp/ckpt/test/myregression.ckpt')
```

如要判断模型是否存在，直接指定目录

```python
checkpoint = tf.train.latest_checkpoint("./tmp/model/")

saver.restore(sess, checkpoint)
```

### 3.4 命令行参数使用 

* 2、 tf.app.flags.,在flags有一个FLAGS标志，它在程序中可以调用到我们

前面具体定义的flag_name

* 3、通过tf.app.run()启动main(argv)函数

```python
# 定义一些常用的命令行参数
# 训练步数
tf.app.flags.DEFINE_integer("max_step", 0, "训练模型的步数")
# 定义模型的路径
tf.app.flags.DEFINE_string("model_dir", " ", "模型保存的路径+模型名字")

# 定义获取命令行参数
FLAGS = tf.app.flags.FLAGS

# 开启训练
# 训练的步数（依据模型大小而定）
for i in range(FLAGS.max_step):
     sess.run(train_op)
```

## 完整代码

```python
# 用tensorflow自实现一个线性回归案例

# 定义一些常用的命令行参数
# 训练步数
tf.app.flags.DEFINE_integer("max_step", 0, "训练模型的步数")
# 定义模型的路径
tf.app.flags.DEFINE_string("model_dir", " ", "模型保存的路径+模型名字")

FLAGS = tf.app.flags.FLAGS

class MyLinearRegression(object):
    """
    自实现线性回归
    """
    def __init__(self):
        pass

    def inputs(self):
        """
        获取特征值目标值数据数据
        :return:
        """
        x_data = tf.random_normal([100, 1], mean=1.0, stddev=1.0, name="x_data")
        y_true = tf.matmul(x_data, [[0.7]]) + 0.8

        return x_data, y_true

    def inference(self, feature):
        """
        根据输入数据建立模型
        :param feature:
        :param label:
        :return:
        """
        with tf.variable_scope("linea_model"):
            # 2、建立回归模型，分析别人的数据的特征数量--->权重数量， 偏置b
            # 由于有梯度下降算法优化，所以一开始给随机的参数，权重和偏置
            # 被优化的参数，必须得使用变量op去定义
            # 变量初始化权重和偏置
            # weight 2维[1, 1]    bias [1]
            # 变量op当中会有trainable参数决定是否训练
            self.weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0),
                                 name="weights")

            self.bias = tf.Variable(0.0, name='biases')

            # 建立回归公式去得出预测结果
            y_predict = tf.matmul(feature, self.weight) + self.bias

        return y_predict

    def loss(self, y_true, y_predict):
        """
        目标值和真实值计算损失
        :return: loss
        """
        # 3、求出我们模型跟真实数据之间的损失
        # 均方误差公式
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

        return loss

    def merge_summary(self, loss):

        # 1、收集张量的值
        tf.summary.scalar("losses", loss)

        tf.summary.histogram("w", self.weight)
        tf.summary.histogram('b', self.bias)

        # 2、合并变量
        merged = tf.summary.merge_all()

        return merged

    def sgd_op(self, loss):
        """
        获取训练OP
        :return:
        """
        # 4、使用梯度下降优化器优化
        # 填充学习率：0 ~ 1    学习率是非常小，
        # 学习率大小决定你到达损失一个步数多少
        # 最小化损失
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        return train_op

    def train(self):
        """
        训练模型
        :param loss:
        :return:
        """

        g = tf.get_default_graph()

        with g.as_default():

            x_data, y_true = self.inputs()

            y_predict = self.inference(x_data)

            loss = self.loss(y_true, y_predict)

            train_op = self.sgd_op(loss)

            # 收集观察的结果值
            merged = self.merge_summary(loss)

            saver = tf.train.Saver()

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                
                # 在没训练，模型的参数值
        		print("初始化的权重：%f, 偏置：%f" % (self.weight.eval(), self.bias.eval()))

                # 加载模型
                checkpoint = tf.train.latest_checkpoint("./tmp/model/")
        		# print(checkpoint)
        		if checkpoint:
            		print('Restoring', checkpoint)
            		saver.restore(sess, checkpoint)
                # 开启训练
                # 训练的步数（依据模型大小而定）
                for i in range(FLAGS.max_step):

                    sess.run(train_op)

                    # 生成事件文件，观察图结构
                    file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)

                    print("训练第%d步之后的损失:%f, 权重：%f, 偏置：%f" % (
                        i,
                        loss.eval(),
                        self.weight.eval(),
                        self.bias.eval()))

                    # 运行收集变量的结果
                    summary = sess.run(merged)

                    # 添加到文件
                    file_writer.add_summary(summary, i)

                    if i % 100 == 0:
                        # 保存的是会话当中的变量op值，其他op定义的值不保存
                        saver.save(sess, FLAGS.model_dir)


if __name__ == '__main__':
    lr = MyLinearRegression()
    lr.train()
```

