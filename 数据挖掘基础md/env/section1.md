# 库的安装

整个数据挖掘基础阶段会用到Matplotlib、Numpy、Pandas、Ta-Lib等库，为了统一版本号在环境中使用，将所有的库及其版本放到了文件requirements.txt当中，然后统一安装

**新建一个用于人工智能环境的虚拟环境**

```
mkvirtualenv -p /user/local/bin/python3 ai
```

```python
matplotlib==2.0.2
numpy==1.14.2
pandas==0.20.3
TA-Lib==0.4.16
tables==3.4.2
jupyter==1.0.0
```
Ta-Lib需要先安装依赖库
```python
安装依赖项
ubuntu:sudo ap-get install ta-lib
maxos:brew install ta-lib
```

# 安装库
pip install TA-Lib

使用pip命令安装

```
pip install -r requirements.txt
```

