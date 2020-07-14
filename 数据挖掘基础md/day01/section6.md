# 其它功能

## 学习目标

- 目标
  - 知道annotate或者text添加图的注释
  - 知道animation实现动画的创建
- 应用
  - 无

## 一、使用annotate和text添加图的注释

```python
fig, ax = plt.subplots(nrows=1, ncols=1, dpi=80)

# 使用splines以及设置颜色，将上方和右方的坐标去除
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 将刻度设置为空，去除刻度
plt.xticks([])
plt.yticks([])

# x,y数据
data = np.ones(100)
data[70:] = list(range(1, 31))
print(data)

# 使用annptate添加注释
plt.annotate(
    '这是一个拐点',
    xy=(70, 1), # 箭头指向位置
    arrowprops=dict(arrowstyle='->'),#自定义箭头样式 
    xytext=(50, 10))# 文本位置

plt.plot(data)

plt.xlabel('1')
plt.ylabel('2')
ax.text(
    30, 2,# 文本位置
    '这是一段文本')
```

效果：

![图形添加文本](/images/图形添加文本.png)

## 二、使用animation实现动画

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

# 设置x,y数据，显示到图形当中
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def init():
    """
    初始设置
    """
    line.set_ydata([np.nan] * len(x))
    return line,


def animate(i):
    """
    更新坐标点函数
    """
    line.set_ydata(np.sin(x + i / 100))
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=True, save_count=50)

plt.show()
```

## 三、更多画图功能

请参考：https://matplotlib.org/tutorials/index.html