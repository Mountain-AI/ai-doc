# 属性

## 学习目标

- 目标
  - 说明数组的属性，形状、类型
- 应用
  - 无

## 一、ndarray

```python
NumPy provides an N-dimensional array type, the ndarray, which describes a collection of “items” of the same type.
```

NumPy提供了一个**N维数组类型ndarray**，它描述了**相同类型**的“items”的集合。

![学生成绩数据](/images/学生成绩数据.png)

### 1、特点

* 每个item都占用相同大小的内存块
* 每个item是由单独的数据类型对象指定的，除了基本类型（整数，浮点数 *等*）之外，**数据类型对象还可以表示数据结构。**

###2、属性

数组属性反映了数组本身固有的信息。

|     属性名字     |          属性解释          |
| :--------------: | :------------------------: |
|  ndarray.shape   |       数组维度的元组       |
|  ndarray.flags   |   有关阵列内存布局的信息   |
|   ndarray.ndim   |          数组维数          |
|   ndarray.size   |      数组中的元素数量      |
| ndarray.itemsize | 一个数组元素的长度（字节） |
|  ndarray.nbytes  |   数组元素消耗的总字节数   |

首先创建一些数组，关于创建数组后详细介绍。

```python
# 创建一个数组
>>> a = np.array([[1,2,3],[4,5,6]])
>>> b = np.array([1,2,3,4])
>>> c = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
```

打印出属性的值

```python
# 类型，大小，字节数
>>> a.dtype # dtype('int64')
>>> a.size # 元素的个数 6
>>> a.nbytes # 总字节数 48
>>> a.itemsize # 一个元素的长度

# 形状比较
# 1维形状 (4,)
# 2维形状 (2,3)
# 3维形状 (2, 2, 3)
>>> a.shape
>>> b.shape
>>> c.shape

(2, 3)
(4,)
(2, 2, 3)

# 内存风格
# 默认C风格
>>> a.flags
C_CONTIGUOUS : True
F_CONTIGUOUS : False
OWNDATA : True
WRITEABLE : True
ALIGNED : True
WRITEBACKIFCOPY : False
UPDATEIFCOPY : False
```

## 3、数组的形状

从刚才打印的形状看到numpy数组的形状表示，那个形状怎么理解。我们可以通过图示的方式表示：

二维数组：

![数组1](/images/数组1.png)

三维数组：

![数组2](/images/数组2.png)



## 4、数组的类型

```python
>>> x = [[0, 1],
       [2, 3]]
>>> x
array([[0, 1],
       [2, 3]])
>>> x.dtype
dtype('int32')
>>> type(x.dtype)
<type 'numpy.dtype'>
```

dtype是numpy.dtype类型，先看看对于数组来说都有哪些类型

|     名称      |                       描述                        | 简写  |
| :-----------: | :-----------------------------------------------: | :---: |
|    np.bool    |      用一个字节存储的布尔类型（True或False）      |  'b'  |
|    np.int8    |             一个字节大小，-128 至 127             |  'i'  |
|   np.int16    |               整数，-32768 至 32767               | 'i2'  |
|   np.int32    |           整数，-2 ** 31 至 2 ** 32 -1            | 'i4'  |
|   np.int64    |           整数，-2 ** 63 至 2 ** 63 - 1           | 'i8'  |
|   np.uint8    |               无符号整数，0 至 255                |  'u'  |
|   np.uint16   |              无符号整数，0 至 65535               | 'u2'  |
|   np.uint32   |           无符号整数，0 至 2 ** 32 - 1            | 'u4'  |
|   np.uint64   |           无符号整数，0 至 2 ** 64 - 1            | 'u8'  |
|  np.float16   | 半精度浮点数：16位，正负号1位，指数5位，精度10位  | 'f2'  |
|  np.float32   | 单精度浮点数：32位，正负号1位，指数8位，精度23位  | 'f4'  |
|  np.float64   | 双精度浮点数：64位，正负号1位，指数11位，精度52位 | 'f8'  |
| np.complex64  |     复数，分别用两个32位浮点数表示实部和虚部      | 'c8'  |
| np.complex128 |     复数，分别用两个64位浮点数表示实部和虚部      | 'c16' |
|  np.object_   |                    python对象                     |  'O'  |
|  np.string_   |                      字符串                       |  'S'  |
|  np.unicode_  |                    unicode类型                    |  'U'  |

#### 4.1创建数组的时候指定类型

```python
>>> a = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
>>> a.dtype
dtype('float32')

>>> arr = np.array(['python','tensorflow','scikit-learn','numpy'],dtype = np.string_)
>>> arr.dtype
array([b'python', b'tensorflow', b'scikit-learn', b'numpy'], dtype='|S12')
```

#### 4.2拓展-自定义数据结构

通常对于numpy数组来说，存储的都是同一类型的数据。但其实也可以通过np.dtype实现 **数据类型对象表示数据结构**。

假设我们现在要存储若干个学生的姓名和身高，那么需要自己定义数据结构实现

> 拓展内容：
>
> ```python
> >>> mytype = np.dtype([('name', np.string_, 10), ('height', np.float64)])
> >>> mytype
> dtype([('name', 'S10'), ('height', '<f8')])
>
> >>> arr = np.array([('Sarah', (8.0)), ('John', (6.0))], dtype=mytype)
> >>> arr
> array([(b'Sarah', 8.), (b'John', 6.)],
>       dtype=[('name', 'S10'), ('height', '<f8')])
> >>> arr[0]['name']
> ```
>
> 对于存储复杂关系的数据，我们其实会选择Pandas更加方便的工具，后面我们详细介绍！



## 二、总结

知道数组的基本属性，不同形状的维度表示以及数组的类型