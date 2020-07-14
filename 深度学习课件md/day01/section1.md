# TF数据流图

## 学习目标

- 目标
  - 说明TensorFlow的数据流图结构

- 应用
  - 无

## 1、数据流图介绍

![数据流图](/images/数据流图.png)

![tensors_flowing](/images/tensors_flowing.gif)

TensorFlow是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Operation）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。

## 2、案例：TensorFlow实现一个加法运算

### 2.1 代码

```python
# 实现一个加法运算
con_a = tf.constant(3.0)
con_b = tf.constant(4.0)

sum_ = tf.add(con_a, con_b)

with tf.Session() as sess:
    print(sess.run(sum_))
```

> 注意问题：警告指出您的CPU支持AVX运算加速了线性代数计算，即点积，矩阵乘法，卷积等。可以从源代码安装TensorFlow来编译，当然也可以选择关闭
>
> ```python
> import os
> os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
> ```

### 2.2 TensorFlow结构分析

TensorFlow 程序通常被组织成**一个构建图阶段和一个执行图阶段**. 在构建阶段, op 的执行步骤 被描述成一个图. 在执行阶段, 使用会话执行执行图中的 op.

* 图和会话 ：
  * 图：这是 TensorFlow 将计算表示为指令之间的依赖关系的一种表示法
  * 会话：TensorFlow 跨一个或多个本地或远程设备运行数据流图的机制
* 张量：TensorFlow 中的基本数据对象
* 节点：提供图当中执行的操作



