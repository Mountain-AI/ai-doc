# 每日作业

### 1、实现sigmoid function,  利用np.exp()

![sigmoid](/images/sigmoid.png)

```python
import numpy as np
def sigmoid(x):
    """
    定义一个sigmoid函数
    x is input: An array
    This function is to compute the sigmoid function value
    """
    s = 1.0 / (1 + (1/np.exp(x)))
    return s
m = np.array([1,2,3])
print(sigmoid(m))
```

