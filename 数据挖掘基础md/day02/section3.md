# 基本操作

##学习目标

- 目标
  - 理解数组的各种创建方法
  - 应用数组的索引机制实现数组的切片获取
  - 应用维度变换实现数组的形状改变
  - 应用类型变换实现数组类型改变
  - 应用数组的转置
  - 说明数组的copy作用
  - 说明正态分布的平均值和标准差
- 应用
  - 应用正态分布实现模拟股票的涨跌幅数据操作

## 一、创建数组

### 1、0和1的数组

* empty(shape[, dtype, order])
* empty_like(a[, dtype, order, subok])	
* eye(N[, M, k, dtype, order])	
* identity(n[, dtype])
* **ones(shape[, dtype, order])**
* ones_like(a[, dtype, order, subok])
* **zeros(shape[, dtype, order])**
* zeros_like(a[, dtype, order, subok])	
* full(shape, fill_value[, dtype, order])	
* full_like(a, fill_value[, dtype, order, subok])

```python
>>> zero = np.zeros([3, 4])
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
```

### 2、从现有的数据中创建

* **array(object[, dtype, copy, order, subok, ndmin])**

- **asarray(a[, dtype, order])**
- asanyarray(a[, dtype, order])
- ascontiguousarray(a[, dtype])	
- asmatrix(data[, dtype])
- **copy(a[, order])**

```python
a = np.array([[1,2,3],[4,5,6]])
# 从现有的数组当中创建
a1 = np.array(a)
# 相当于索引的形式，并没有真正的创建一个新的
a2 = np.asarray(a)
```

#### 2.1、关于array和asarray的不同

![asarray](/images/asarray.png)

### 3、创建固定范围的数组

* np.linspace (start, stop, num, endpoint, retstep, dtype)

生成等间隔的序列

```
start 序列的起始值
stop 序列的终止值，
如果endpoint为true，该值包含于序列中
num 要生成的等间隔样例数量，默认为50
endpoint 序列中是否包含stop值，默认为ture
retstep 如果为true，返回样例，
以及连续数字之间的步长
dtype 输出ndarray的数据类型
```

```python
# 生成等间隔的数组
np.linspace(0, 100, 10)
```

* 其它的还有
  * numpy.arange(start,stop, step, dtype)
  * numpy.logspace(start,stop, num, endpoint, base, dtype)

```python
np.arange(10, 50, 2)
```

### 3、创建随机数组

* np.random模块
  * 均匀分布
    * np.random.rand(10)
    * np.random.uniform(0,100)
    * np.random.randint(100)
  * 正态分布？
    * 给定均值／标准差／维度的正态分布
    * np.random.normal(1.75, 0.2, (3,4))
    * np.random.standard_normal(size=(3,4))

```python
# 创建均匀分布的数组
# 0~1
np.random.rand(10)

# 默认范围一个数
np.random.uniform(0, 100)

# 随机整数
np.random.randint(10)

np.random.normal(1.75, 0.1, (10, 10))
```

## 二、正态分布（理解）

### 1、什么是正态分布

正态分布是一种概率分布。正态分布是具有两个参数μ和σ的连续型随机变量的分布，第一参数μ是服从正态分布的随机变量的均值，第二个参数σ是此随机变量的方差，所以正态分布记作**N(μ，σ )**。

![正态分布](/images/正态分布.png)

### 2、正态分布的应用

**生活、生产与科学实验中很多随机变量的概率分布都可以近似地用正态分布来描述。**

### 3、正态分布特点

**μ决定了其位置，其标准差σ**。决定了分布的幅度。当μ = 0,σ = 1时的正态分布是标准正态分布。

![正态分布公式](/images/正态分布公式.png)

标准差如何来？

#### 3.1方差

是在概率论和统计方差衡量一组数据时离散程度的度量

![方差公式](/images/方差公式.png)

其中M为平均值，n为数据总个数，S为标准差，S^2可以理解一个整体为方差

![标准差公式](/images/标准差公式.png)

### 标准差与方差的意义

可以理解成数据的一个离散程度的衡量

![离散程度](/images/离散程度.png)

* 有了方差为什么需要标准差去衡量？



例如：我们可以模拟生成一组股票的涨跌幅的数据

## 三、案例：随机生成500个股票两年的交易日涨幅数据

500只股票，**两年(504天)**的涨跌幅数据，如何获取？

* 两年的交易日数量为：2 X 252 = 504
* 随机生成涨跌幅在某个正态分布内，比如均值0，方差1

### 1、股票涨跌幅数据的创建

```python
# 创建一个符合正太分布的500个股票504天的涨跌幅数据
stock_day_rise = np.random.normal(0, 1, (500, 504))
stock_day_rise.shape
```

###2、数组的索引

* 获取第一个股票的前100个交易日的涨跌幅数据

```python
# 二维的数组，两个维度 
stock_day_rise[0, 0:10]
```

一维、二维、三维的数组如何索引？

```python
# 三维，一维
a1 = np.array([ [[1,2,3],[4,5,6]], [[12,3,34],[5,6,7]]])
a1[0, 0, 1]
```

###3、数组形状与类型变化

####3.1修改形状

让刚才的股票行、日期列反过来，变成日期行，股票列

* **ndarray.reshape(shape[, order])	Returns an array containing the same data with a new shape. **

```python
# 在转换形状的时候，一定要注意数组的元素匹配
stock_day_rise.reshape([504, 500])
```

* **ndarray.resize(new_shape[, refcheck])	Change shape and size of array in-place.**

```python
stock_day_rise.resize([504,500])
```

* **ndarray.flatten([order])	Return a copy of the array collapsed into one dimension.**

```python
stock_day_rise.flatten()
```

#### 3.2修改类型

* **ndarray.astype(type)**

```python
stock_day_rise.reshape([504, 500]).astype(np.int32)
```

#### 3.3修改小数位数

* **ndarray.round(arr, out)	Return a with each element rounded to the given number of decimals.**

```python
np.round(stock_day_rise[:2, :20], 4)
```

###4、数组转换

* **ndarray.T	数组的转置**
  * 将数组的行、列进行互换

```python
stock_day_rise.shape
(500, 504)
stock_day_rise.T.shape
(504, 500)
```

* **ndarray.tostring([order])或者ndarray.tobytes([order])  Construct Python bytes containing the raw data bytes in the array.**
  * 转换成bytes

```python
arr = np.array([ [[1,2,3],[4,5,6]], [[12,3,34],[5,6,7]]])
arr.tostring()
```

如果遇到：

```
IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
```

这个问题是在jupyer当中对输出的字节数有限制，需要去修改配置文件

创建配置文件

```python
jupyter notebook --generate-config
vi ~/.jupyter/jupyter_notebook_config.py
```

取消注释,多增加

```python
## (bytes/sec) Maximum rate at which messages can be sent on iopub before they
#  are limited.
c.NotebookApp.iopub_data_rate_limit = 10000000
```

**但是不建议这样去修改，jupyter输出太大会崩溃**

* **ndarray.copy([order])	Return a copy of the array.**

```python
# 先从两年stock_day_rise拷贝一些数据
temp = stock_day_rise[:4, :4].copy()
```
当我们不想修改某个股票数据的时候，就可以去进行拷贝操作。在拷贝的数据上进行操作

## 四、小结

* 创建数组
  * 均匀
  * **随机（正态分布）**
* 正态分布
* 数组索引
* 数组形状改变
* 数组类型
  * reshape
  * resize
  * flatten
* 数组转换
  * T
  * tostring
  * copy