# 合并、分割

## 学习目标

- 目标
  - 应用concatenate、vstack、hstack实现数组合并
  - 应用split实现数组的横、纵向分割
- 应用
  - 无

## 一、合并、分割的用处

实现数据的切分和合并，将数据进行切分合并处理

## 二、合并

* numpy.concatenate((a1, a2, ...), axis=0)
* numpy.hstack(tup) Stack arrays in sequence horizontally **(column wise).**
* numpy.vstack(tup) Stack arrays in sequence vertically **(row wise).**

比如我们将两部分股票的数据拼接在一起：

```python
a = stock_day_rise[:2, 0:4]
b = stock_day_rise[10:12, 0:4]

# axis=1时候，按照数组的列方向拼接在一起
# axis=0时候，按照数组的行方向拼接在一起
np.concatenate([a, b], axis=0)

array([[-2.59680892, -2.44345152, -2.15348934, -1.86554389],
       [-1.04230807,  1.33132386,  0.52063143,  0.49936452],
       [-1.3083418 , -1.08059664,  0.60855154,  0.1262362 ],
       [ 0.87602641,  0.07077588, -0.44194904,  0.87074559]])

np.hstack([a,b])
np.vstack([a,b])
```

## 三、分割

* **numpy.split(ary, indices_or_sections, axis=0)  Split an array into multiple sub-arrays.**

```python
np.split(ab, 4, axis=0)
```

