# 变量OP

- 目标
  - 说明变量op的特殊作用
  - 说明变量op的trainable参数的作用
  - 应用global_variables_initializer实现变量op的初始化
- 应用
  - 无

## 1、变量

TensorFlow变量是表示程序处理的共享持久状态的最佳方法。变量通过 tf.Variable OP类以及tf.get_variable()类进行操作。变量的特点

* **存储持久化**
* **可修改值**
* **可指定被训练**

### 1.1创建变量

* tf.Variable(**initial_value=None,trainable=True,collections=None**,name=None)
  * initial_value:初始化的值
  * trainable:是否被训练
  * collections：新变量将添加到列出的图的集合中collections，默认为[GraphKeys.GLOBAL_VARIABLES]，如果trainable是True变量也被添加到图形集合 GraphKeys.TRAINABLE_VARIABLES 

```python
var = tf.Variable(tf.random_normal([2, 2], mean=0.0, stddev=1.0), name="var", trainable=True)

with tf.Session() as sess:
    sess.run(var)
```

* 变量需要显示初始化，才能运行值

```python
# 添加一个初始化变量的OP
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行初始化变量的OP
    sess.run(init_op)

```

### 1.2 变量OP的方法

* new_var = assign(value)
  * 给变量赋值一个新的值
* new_var = assign_add(delta)

```python
var = tf.Variable(tf.random_normal([2, 2], mean=0.0, stddev=1.0), name="var", trainable=True)
var1 = var.assign([[2, 3], [4, 5]])
init_op = tf.global_variables_initializer()
va = var.assign_add([[1, 3], [4, 5]])
with tf.Session() as sess:
    # 运行初始化op
    sess.run(init_op)
    print(sess.run(va))
    print(sess.run(var))
```

关于变量的被训练，我们在后面的线性回归案例当中介绍

## 2、命名空间与共享变量

**共享变量的主要用途在一些网络当中的参数共享**， 由于在TensorFlow当中，只要我们定义的OP,name参数指定一样，其实并不是同一个变量。如果想要达到重复利用变量的效果，我们就要使用`tf.variable_scope()`结合`tf.get_variable()`一起使用

### 2.1 定义一个相同名字的变量

```python
var = tf.Variable(name='var', initial_value=[4], dtype=tf.float32)
var_double = tf.Variable(name='var', initial_value=[4], dtype=tf.float32)

<tf.Variable 'var:0' shape=() dtype=float32_ref>
<tf.Variable 'var_1:0' shape=() dtype=float32_ref>
```

### 2.2 使用tf.variable_scope()修改OP命名空间

会在OP的名字前面增加命名空间的指定名字

```python
with tf.variable_scope("name"):
    var = tf.Variable(name='var', initial_value=[4], dtype=tf.float32)
    var_double = tf.Variable(name='var', initial_value=[4], dtype=tf.float32)
    
<tf.Variable 'name/var:0' shape=() dtype=float32_ref>
<tf.Variable 'name/var_1:0' shape=() dtype=float32_ref>
```

### 2.2 tf.get_variable共享变量

通过tf.get_variable的初始化与Variable参数一样,**但是要是实现共享需要打开tf.variable_scope("name")中的reuse=tf.AUTO_REUSE参数** 


```python
# 打开共享参数
# 或者
#  with tf.variable_scope("name") as scope:
#  在需要使用共享变量的前面定义： scope.reuse_variables()
with tf.variable_scope("name", reuse=tf.AUTO_REUSE):
    var = tf.Variable(initial_value=4.0, name="var", dtype=tf.float32)
    var_double = tf.Variable(initial_value=4.0, name="var", dtype=tf.float32)

    var1 = tf.get_variable(initializer=tf.random_normal([2, 2], mean=0.0, stddev=1.0),
                           name="var1",
                           dtype=tf.float32)
    var1_double = tf.get_variable(initializer=tf.random_normal([2, 2], mean=0.0, stddev=1.0),
                           name="var1",
                           dtype=tf.float32)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1)
    print(var1_double)
```





