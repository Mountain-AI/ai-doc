# 统计运算

## 学习目标

- 目标
  - 使用np.max完成最大值计算
  - 使用np.min完成最小值计算
  - 使用np.mean完成平均值计算
  - 使用np.std完成标准差计算
  - 使用np.argmax、np.argmin完成最大值最小值的索引
- 应用
  - 股票涨跌幅数据统计

## 一、问题？

为什么需要统计运算？

## 二、统计指标

在数据挖掘/机器学习领域，统计指标的值也是我们分析问题的一种方式。常用的指标如下：

* min(a[, axis, out, keepdims])    Return the minimum of an array or minimum along an axis.
* max(a[, axis, out, keepdims])	Return the maximum of an array or maximum along an axis.

- median(a[, axis, out, overwrite_input, keepdims])	Compute the median along the specified axis.
- mean(a[, axis, dtype, out, keepdims])	Compute the arithmetic mean along the specified axis.
- std(a[, axis, dtype, out, ddof, keepdims])	Compute the standard deviation along the specified axis.
- var(a[, axis, dtype, out, ddof, keepdims])	Compute the variance along the specified axis.



## 三、股票涨跌幅统计运算

进行统计的时候，**axis 轴的取值并不一定，Numpy中不同的API轴的值都不一样，在这里，axis 0代表列,  axis 1代表行去进行统计**

```python
# 接下来对于这4只股票的4天数据，进行一些统计运算
# 指定行 去统计
print("所有四只股票前四天的最大涨幅{}".format(np.max(temp, axis=1)))
# 使用min, std, mean
print("所有四只股票前100天的最大跌幅{}".format(np.min(temp, axis=1)))
print("所有四只股票前100天的振幅幅度{}".format(np.std(temp, axis=1)))
print("所有四只股票前100天的平均涨跌幅{}".format(np.mean(temp, axis=1)))
```

如果需要统计出哪一只股票在某个交易日的涨幅最大或者最小？

* np.argmax(temp, axis=)
* np.argmin(temp, axis=)

```python
# 获取股票指定哪一天的涨幅最大
print("前四只股票在100天内涨幅最大{}".format(np.argmax(temp, axis=1)))
print("前100天在天内涨幅最大的股票{}".format(np.argmax(temp, axis=0)))
```

