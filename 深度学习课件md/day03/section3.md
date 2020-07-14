# 案例：CNN-Mnist手写数字识别

## 学习目标

- 目标
  - 应用tf.nn.conv2d实现卷积计算
  - 应用tf.nn.relu实现激活函数计算
  - 应用tf.nn.max_pool实现池化层的计算
  - 应用卷积神经网路实现图像分类识别
- 应用
  - CNN-Mnist手写数字识别

## 1、网络设计

我们自己定义一个卷积神经网络去做识别，这里定义的结构有些是通常大家都会采用的数量以及熟练整个网络计算流程。但是至于怎么定义结构是没办法确定的，也就是神经网络的黑盒子特性，如果想自己设计网络通常还是比较困难的，可以使用一些现有的网络结构如之前的GoogleNet、VGG等等

### 1.1 网络结构

![卷积网络设计](/images/卷积网络设计.png)

### 1.2 具体参数

* 第一层
  * 卷积：32个filter、大小5*5、strides=1、padding="SAME"
  * 激活：Relu
  * 池化：大小2x2、strides2
* 第一层
  * 卷积：64个filter、大小5*5、strides=1、padding="SAME"
  * 激活：Relu
  * 池化：大小2x2、strides2
* 全连接层

**经过每一层图片数据大小的变化需要确定，Mnist输入的每批次若干图片数据大小为[None, 784]，如果要进过卷积计算，需要变成[None, 28, 28, 1]**

* 第一层
  - 卷积：[None, 28, 28, 1]———>[None, 28, 28, 32]
    - 权重数量：[5, 5, 1 ,32]
    - 偏置数量：[32]
  - 激活：[None, 28, 28, 32]———>[None, 28, 28, 32]
  - 池化：[None, 28, 28, 32]———>[None, 14, 14, 32]
* 第二层
  - 卷积：[None, 14, 14, 32]———>[None, 14, 14, 64]
    - 权重数量：[5, 5, 32 ,64]
    - 偏置数量：[64]
  - 激活：[None, 14, 14, 64]———>[None, 14, 14, 64]
  - 池化：[None, 14, 14, 64]———>[None, 7, 7, 64]
* 全连接层
  * [None, 7, 7, 64]———>[None, 7 * 7 * 64]
  * 权重数量：[7 * 7 * 64, 10]，由分类别数而定
  * 偏置数量：[10]，由分类别数而定

## 2、案例：CNN识别Mnist手写数字

### 2.1 流程

1、准备数据

2、卷积、激活、池化（两层）

3、全连接层

4、计算损失、优化

5、计算准确率

### 2.2 代码

* 网络结构实现

```python
def conv_model():
    """
    自定义的卷积网络结构
    :return: x, y_true, y_predict
    """
    # 1、准备数据占位符
    # x [None, 784]  y_true [None, 10]
    with tf.variable_scope("data"):

        x = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2、卷积层一 32个filter, 大小5*5,strides=1, padding=“SAME”

    with tf.variable_scope("conv1"):
        # 随机初始化这一层卷积权重 [5, 5, 1, 32], 偏置[32]
        w_conv1 = weight_variables([5, 5, 1, 32])

        b_conv1 = bias_variables([32])

        # 首先进行卷积计算
        # x [None, 784]--->[None, 28, 28, 1]  x_conv1 -->[None, 28, 28, 32]
        x_conv1_reshape = tf.reshape(x, [-1, 28, 28, 1])
        # input-->4D
        x_conv1 = tf.nn.conv2d(x_conv1_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1

        # 进行激活函数计算
        #  x_relu1 -->[None, 28, 28, 32]
        x_relu1 = tf.nn.relu(x_conv1)

        # 进行池化层计算
        # 2*2, strides 2
        #  [None, 28, 28, 32]------>[None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3、卷积层二 64个filter, 大小5*5,strides=1,padding=“SAME”
    # 输入：[None, 14, 14, 32]
    with tf.variable_scope("conv2"):
        # 每个filter带32张5*5的观察权重，一共有64个filter去观察
        # 随机初始化这一层卷积权重 [5, 5, 32, 64], 偏置[64]
        w_conv2 = weight_variables([5, 5, 32, 64])

        b_conv2 = bias_variables([64])

        # 首先进行卷积计算
        # x [None, 14, 14, 32]  x_conv2 -->[None, 14, 14, 64]
        # input-->4D
        x_conv2 = tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2

        # 进行激活函数计算
        #  x_relu1 -->[None, 28, 28, 32]
        x_relu2 = tf.nn.relu(x_conv2)

        # 进行池化层计算
        # 2*2, strides 2
        #  [None, 14, 14, 64]------>x_pool2[None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 4、全连接层输出
    # 每个样本输出类别的个数10个结果
    # 输入：x_poll2 = [None, 7, 7, 64]
    # 矩阵运算： [None, 7 * 7 * 64] * [7 * 7 * 64, 10] +[10] = [None, 10]
    with tf.variable_scope("fc"):
        # 确定全连接层权重和偏置
        w_fc = weight_variables([7 * 7 * 64, 10])

        b_fc = bias_variables([10])

        # 对上一层的输出结果的形状进行处理成2维形状
        x_fc = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

        # 进行全连接层运算
        y_predict = tf.matmul(x_fc, w_fc) + b_fc

    return x, y_true, y_predict
```

* 损失计算优化、准确率计算

```python
	# 1、准备数据API
    mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)

    # 2、定义模型,两个卷积层、一个全连接层
    x, y_true, y_predict = conv_model()

    # 3、softmax计算和损失计算
    with tf.variable_scope("softmax_loss"):

        # labels:真实值 [None, 10]  one_hot
        # logits:全脸层的输出[None,10]
        # 返回每个样本的损失组成的列表
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                                      logits=y_predict))
    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        # train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

    # 5、准确率计算
    with tf.variable_scope("accuracy"):

        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量op
    init_op = tf.global_variables_initializer()
```

* 会话运行

```python
# 会话运行
    with tf.Session() as sess:

        # 初始化变量
        sess.run(init_op)

        # 循环去训练模型
        for i in range(2000):

            # 获取数据，实时提供
            # 每步提供50个样本训练
            mnist_x, mnist_y = mnist.train.next_batch(50)

            sess.run(train_op, feed_dict={x:mnist_x, y_true: mnist_y})

            # 打印准确率大小
            print("第%d步训练的准确率为:--%f" % (i,
                                        sess.run(accuracy, feed_dict={x:mnist_x, y_true: mnist_y})
                                        ))
```

## 2.3 学习率过大问题

**发现当我们设置0.1的学习率之后，准确率一直上不去，并且打印参数发现已经变为NaN，这个地方是不是与之前在做线性回归的时候似曾相识。对于卷积网络来说，更容易发生梯度爆炸现象，只能通过调节学习来避免。**

## 3、拓展-Tensorflow高级API实现结构

高级API可以更快的构建模型，但是对神经网络的运行流程了解清晰还是需要去使用底层API去构建模型，更好理解网络的原理

https://www.tensorflow.org/tutorials/layers

```python
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # 全连接层计算+dropout
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # 得出预测结果
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # 计算损失
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # 配置train_op
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # 评估模型
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

