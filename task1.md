## 随机数种子设为1时，打出1-100间random随机数输出的前20个
```python
import numpy as np

np.random.seed(1)
X = np.random.randint(1,100,size=(5,20))
Y = X[0,:]
print(X)
print(Y)

np.random.seed(5)
A = np.random.randint(1,100,size=(5,20))
B = A[0,:]
print(A)
print(B)
```
## 读取ucf101_01.json，统计里面各个层级key是什么
## 统计training set和validation set视频个数
## 各类label在training set和validation set下各有多少个视频
```python
import json
import random
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
with open ('/home/liqianqian/Downloads/ucf101_list/ucf101_01.json','r') as u_file:
    ucf101_dir = json.loads(u_file.read())
    a = type(ucf101_dir)
    print(a)
    #print (ucf101_dir)
for key in ucf101_dir.keys():
    data1 = ucf101_dir[key] #将当前字典层级的key赋给data
    if isinstance(data1,dict):
        for key in data1.keys():#data1是个字典，那么遍历data1的所有key
            data2=data1[key]    #把data1的键给data2
            list1.append(key)
            if isinstance(data2,dict):
                for key1 in data2.keys():
                    data3=data2[key1]
                    list2.append(key1)
                    if isinstance(data3,dict):
                        for key2,value2 in data3.items():
                            list3.append(key2)
                            list5.append(value2)
                    else: list4.append(data3)
                    i = 0
                    j = 0
                    for A in list4:
                        if A == "training":
                            i = i+1
                        if A == "validation":
                            j = j+1
print(list3)
print(list2)
print(list1)
print(list4)
print(list5)
print(i)      #training个数
print(j)      #validation个数


c = {}
d = {}
i = 0
j = 0
for videos in list1:
    a = ucf101_dir['database'][videos]['annotations']['label']
    c[a] = 0
    d[a] = 0
for videos in list1:
    if ucf101_dir['database'][videos]['subset'] == 'training':
        a = ucf101_dir['database'][videos]['annotations']['label']
    c[a]+=1
    if ucf101_dir['database'][videos]['subset'] == 'validation':
        a = ucf101_dir['database'][videos]['annotations']['label']
    d[a]+=1
print(c)
print(d)
```
## 如何抽100个左右视频做个小的数据集训练测试
```python
m = {}
l = {}
i=0
k = ucf101_dir['labels']   #把labels引进新的字典m中
m['labels'] = k[:]
for key3 in ucf101_dir['database'].keys():
    t = random.sample(ucf101_dir['database'].keys(),100)   #t是列表
for u in t:                  #u是键
    g = ucf101_dir['database'][u]             #g是值
    l[u] = g
    i+=1
m['database'] = l
print(m)      #小数据集
print(i)      #统计视频个数
```




