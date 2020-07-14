# 柱状图

## 学习目标

- 目标
  - 应用bar实现柱状图的绘制
  - 知道柱状图的应用场景
- 应用
  - 电影票房收入绘制



matplotlib能够绘制**折线图、柱状图、饼图、直方图、**散点图、热力图、K线图等，但是,我们需要知道不同的统计图到底能够表示出什么,以此来决定选择哪种统计图来更直观的呈现我们的数据

## 一、常见图形种类及意义

![图形种类](/images/图形种类.png)

* 折线图:以折线的上升或下降来表示统计数量的增减变化的统计图

  **特点:能够显示数据的变化趋势，反映事物的变化情况。(变化)**

* 直方图:由一系列高度不等的纵向条纹或线段表示数据分布的情况。 一般用横轴表示数据范围，纵轴表示分布情况。

  **特点:绘制,连续性的数据展示一组或者多组数据的分布状况(统计)**

* 柱状图:排列在工作表的列或行中的数据可以绘制到柱状图中。

  **特点:绘制连离散的数据,能够一眼看出各个数据的大小,比较数据之间的差别。(统计)**

* 散点图:用两组数据构成多个坐标点，考察坐标点的分布,判断两变量之间是否存在某种关联或总结坐标点的分布模式。

  **特点:判断变量之间是否存在数量关联趋势,展示离群点(分布规律)**

## 二、柱状图图绘制

![柱状图](/images/柱状图.png)

##需求：每部电影的票房收入对比？

### 1、画出每部电影的票房收入对比，效果如下：

![电影票房条形图](/images/电影票房条形图.png)

电影数据如下图所示：

![电影票房数据](/images/电影票房数据.png)

```
['雷神3：诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴', '降魔传','追捕','七十七天','密战','狂兽','其它']
[73853,57767,22354,15969,14839,8725,8716,8318,7916,6764,52222]
```

### 2、绘制

* matplotlib.pyplot.bar(x, width, align='center', **kwargs)

绘制柱状图

```

Parameters:	
x : sequence of scalars.

width : scalar or array-like, optional
柱状图的宽度

align : {‘center’, ‘edge’}, optional, default: ‘center’
Alignment of the bars to the x coordinates:
‘center’: Center the base on the x positions.
‘edge’: Align the left edges of the bars with the x positions.
每个柱状图的位置对齐方式

**kwargs :
color:选择柱状图的颜色

Returns:	
`.BarContainer`
Container with all the bars and optionally errorbars.
```

代码：

```python
# 完成简单的条形图展现不同的电影票房之间的对比
plt.figure(figsize=(20, 8), dpi=80)

# 准备电影的名字以及电影的票房数据
movie_name = ['雷神3：诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴','降魔传','追捕','七十七天','密战','狂兽','其它']
y = [73853,57767,22354,15969,14839,8725,8716,8318,7916,6764,52222]
# 放进横坐标的数字列表
x = range(len(movie_name))

# 画出条形图
plt.bar(x, y, width=0.5, color=['b','r','g','y','c','m','y','k','c','g','g'])

# 修改刻度名称
plt.xticks(x, movie_name)

plt.show()
```

## 如何对比电影票房收入才更能加有说服力？

### 3、比较相同天数的票房

有时候为了公平起见，我们需要对比不同电影首日和首周的票房

####3.1数据如下

```
movie_name = ['雷神3：诸神黄昏','正义联盟','寻梦环游记']

first_day = [10587.6,10062.5,1275.7]
first_weekend=[36224.9,34479.6,11830]

数据来源: https://piaofang.maoyan.com/?ver=normal
```

效果如下：

![首日首周票房对比](/images/首日首周票房对比.png)

#### 3.2 分析

* 添加首日首周两部分的柱状图
* x轴中文坐标位置调整

代码：

```python
# 三部电影的首日和首周票房对比
plt.figure(figsize=(20, 8), dpi=80)

movie_name = ['雷神3：诸神黄昏','正义联盟','寻梦环游记']

first_day = [10587.6,10062.5,1275.7]
first_weekend=[36224.9,34479.6,11830]

x = range(len(movie_name))

# 画出柱状图
plt.bar(x, first_day, width=0.2, label="首日票房")
# 首周柱状图显示的位置在首日的位置右边
plt.bar([i+0.2 for i in x], first_weekend, width=0.2, label="首周票房")

# 显示X轴中文，固定在首日和首周的中间位置
plt.xticks([i+0.1 for i in x], movie_name)
plt.legend(loc='best')

plt.show()
```



## 三、柱状图应用场景

适合用在分类数据对比场景上

* 数量统计
* 用户数量对比分析

