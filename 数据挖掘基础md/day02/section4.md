# 逻辑运算

## 学习目标

- 目标
  - 应用数组的通用判断函数
  - 应用np.where实现数组的三元运算
- 应用
  - 股票涨跌幅数据逻辑运算

## 一、问题？

**如果我们想要判断获取涨幅大于0.5一写区段？**

## 二、逻辑运算

```python
# 逻辑判断
temp > 0.5

# 赋值
temp[temp > 0.5] = 1
```

## 三、通用判断函数

* np.all()

```python
#判断stock_day_rise[0:2,0:5]是否全是上涨的
np.all(stock_day_rise[0:2,0:5] > 0)
```

* np.unique()

返回新的数组的数值，不存在重复的值

```python
#将序列中数值值唯一且不重复的值组成新的序列

change_int = stock_day_rise[0:2,0:5].astype(int)
np.unique(change_int)
```

## 四、np.where（三元运算符）

通过使用np.where能够进行更加复杂的运算

* np.where()

```python
np.where(temp > 0, 1, 0)
```

* 复合逻辑需要结合np.logical_and和np.logical_or使用

```python
# 判断前四个股票前四天的涨跌幅 大于0.5并且小于1的，换为1，否则为0
# 判断前四个股票前四天的涨跌幅 大于0.5或者小于-0.5的，换为1，否则为0
np.where(np.logical_and(temp > 0.5, temp < 1), 1, 0)
np.where(np.logical_or(temp > 0.5, temp < -0.5), 1, 0)
```



