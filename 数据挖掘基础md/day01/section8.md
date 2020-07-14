# 每日作业

### 一、实现如下饼图

* 增加阴影
* 增加破裂效果

![作业1](/images/作业1.png)

```python
import matplotlib.pyplot as plt


labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()
```

## 二、实现K线图绘制

* 使用

  ```
  matplotlib.finance.candlestick_ochl(ax, quotes, width=0.2, colorup='k', colordown='r')
  ```

效果：

![作业2](/images/作业2.png)

```python
# 数据的获取和处理
data = pd.read_hdf("./data/stock_plot/day_open.h5")[:100]
data1 = pd.read_hdf("./data/stock_plot/day_close.h5")[:100]
data2 = pd.read_hdf("./data/stock_plot/day_high.h5")[:100]
data3 = pd.read_hdf("./data/stock_plot/day_low.h5")[:100]

day = pd.concat([data["000001.SZ"], data1["000001.SZ"], data2["000001.SZ"],
data3["000001.SZ"]], axis=1)

day.columns = ["open", "close", "high", "low"]
day = day.reset_index().values

# 画图
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 8), dpi=80)
# 第一个参数axes
candlestick_ochl(axes, day, width=0.2, colorup='r', colordown='g')
plt.show()
```
参考API：https://matplotlib.org/api/finance_api.html?highlight=candlestick#matplotlib.finance.candlestick_ochl

https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes