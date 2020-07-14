# 高级处理-缺失值处理

## 学习目标

- 目标 
  - 说明Pandas的缺失值类型
  - 应用replace实现数据的替换
  - 应用dropna实现缺失值的删除
  - 应用fillna实现缺失值的填充
  - 应用isnull判断是否有缺失数据NaN
- 应用
  - 对电影数据进行缺失值处理

###如何处理缺失值？

![缺失值](/images/缺失值.png)

## 1、电影数据的缺失值处理

### 1.1 电影数据文件获取

```python
# 读取电影数据
movie = pd.read_csv("./data/IMDB/IMDB-Movie-Data.csv")
```

### 1.2 缺失值的处理逻辑

对于NaN的数据，在numpy中我们是如何处理的？在pandas中我们处理起来非常容易

* 判断数据是否为NaN：pd.isnull(df),pd.notnull(df)

处理方式：

* 存在缺失值nan,并且是np.nan:
  * 1、删除存在缺失值的:dropna(axis='rows')
  * 2、替换缺失值:fillna(df[].mean(), inplace=True)
* 不是缺失值nan，有默认标记的

### 1.3 存在缺失值nan,并且是np.nan

* 删除

```python
# pandas删除缺失值，使用dropna的前提是，缺失值的类型必须是np.nan
movie.dropna()
```

* 替换缺失值

```python
# 替换存在缺失值的样本
# 替换？  填充平均值，中位数
movie['Revenue (Millions)'].fillna(movie['Revenue (Millions)'].mean(), inplace=True)

movie['Metascore'].fillna(movie['Metascore'].mean(), inplace=True)
```

### 1.4 不是缺失值nan，有默认标记的

数据是这样的：

![问号缺失值](/images/问号缺失值.png)

```python
wis = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
```

**处理思路分析：**

* 1、先替换‘?’为np.nan

  df.replace(to_replace=, value=)

* 2、在进行缺失值的处理

```python
# 把一些其它值标记的缺失值，替换成np.nan
wis = wis.replace(to_replace='?', value=np.nan)

wis.dropna()
```