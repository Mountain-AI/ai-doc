# IO操作与数据处理

## 学习目标

- 目标
  - 知道Numpy文件的读取
- 应用
  - 无

## 一、问题？

大多数数据并不是我们自己构造的，存在文件当中。我们需要工具去获取，但是Numpy其实并不适合去读取处理数据，这里我们了解相关API，以及Numpy不方便的地方即可。

## 二、Numpy读取

* genfromtxt(fname[, dtype, comments, ...])	Load data from a text file, with missing values handled as specified.

![numpy_file](/images/numpy_file.png)

```python
# 读取数据
test = np.genfromtxt("./data/numpy_test/test.csv", delimiter=',')
```

## 三、如何处理缺失值

#### 3.1什么是缺失值

什么时候numpy中会出现nan：当我们读取本地的文件为float的时候，如果有缺失(或者为None)，就会出现nan

#### 3.2缺失值处理？

那么，在一组数据中单纯的把nan替换为0，合适么？会带来什么样的影响？

比如，全部替换为0后，替换之前的平均值如果大于0，替换之后的均值肯定会变小，所以更一般的方式是把缺失的数值替换为均值（中值）或者是直接删除有缺失值的一行

所以：

* 如何计算一组数据的中值或者是均值
* 如何删除有缺失数据的那一行（列）**在pandas中介绍**



```python
t中存在nan值，如何操作把其中的nan填充为每一列的均值
t = array([[  0.,   1.,   2.,   3.,   4.,   5.],
       [  6.,   7.,  nan,   9.,  10.,  11.],
       [ 12.,  13.,  14.,  nan,  16.,  17.],
       [ 18.,  19.,  20.,  21.,  22.,  23.]])
```

处理逻辑：

![缺失处理逻辑](/images/缺失处理逻辑.png)

看了上面的处理过程，非常麻烦，别担心之后我们会介绍强大的Pandas工具进行处理！！