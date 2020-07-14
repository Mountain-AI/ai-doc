# 逻辑与算数运算

## 学习目标

- 目标
  - 应用逻辑运算符号实现数据的逻辑筛选
  - 应用isin实现数据的筛选
  - 应用add等实现数据间的加法运算
  - 应用apply函数实现数据的自定义处理
- 应用
  - 实现股票的逻辑筛选

#### 主要用于对股票的筛选操作

## 1、使用逻辑运算符号<、>等进行筛选

```python
# 进行逻辑判断
# 用true false进行标记，逻辑判断的结果可以作为筛选的依据
data[data['p_change'] > 2]
```

## 2、使用|、&完成复合的逻辑

```python
# 完成一个符合逻辑判断， p_change > 2, open > 15
data[(data['p_change'] > 2) & (data['open'] > 15)]
```

## 3、isin()

```python
# 可以指定值进行一个判断，从而进行筛选操作
data[data['turnover'].isin([4.19])]
data.head(10)
```

## 4、数学运算

**如果想要得到每天的涨跌大小？**

```python
# 进行数学运算 加上具体的一个数字
data['open'].add(1)

# 自己求出每天 close- open价格差
# 筛选两列数据
close = data['close']
open1 = data['open']
# 默认按照索引对齐
data['m_price_change'] = close.sub(open1)
```

## 5、自定义运算函数

```python
# 进行apply函数运算
data[['open', 'close']].apply(lambda x: x.max() - x.min(), axis=0)
data[['open', 'close']].apply(lambda x: x.max() - x.min(), axis=1)
```

