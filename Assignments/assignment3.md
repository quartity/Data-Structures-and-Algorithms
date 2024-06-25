# Assignment #3: March月考

Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by 王铭健，工学院



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2

Python编程环境：PyCharm 2023.2.1 (Professional Edition)

C/C++编程环境：暂无



## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



【王铭健，工学院，2024年春】用时：20min

思路：

一个很经典的dp问题。题目翻译过来就是“求最长单调递减子序列”，与上学期的“最长上升子序列”做法基本完全一致。

但既然月考把这道题放在了“E”，我想穷举应当也是可以的。

##### 代码

```python
# 王铭健，工学院 2300011118
n = int(input())
heights = list(map(int, input().split()))
dp = [1] * n
for i in range(1, n):
    for j in range(i):
        if heights[j] >= heights[i]:
            dp[i] = max(dp[j] + 1, dp[i])
print(max(dp)) 

```



代码运行截图

![image-20240312175324522](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312175324522.png)



**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



【王铭健，工学院，2024年春】用时：20min

思路：

运用递归写法。若要将n个盘（n>1）从a移到c，就需要把前n-1个盘从a移到b，再把第n个盘从a移到c，最后把前n-1个盘从b移到c。n=1时，直接将盘从a移到c即可。将这个思路用递归函数写出来即可。

##### 代码

```python
# 王铭健，工学院 2300011118
num, a, b, c = input().split()
num = int(num)
cols = {a, b, c}


def move(n, start, end):
    while n > 1:
        rest = list(cols - {start, end})[0]
        move(n-1, start, rest)
        print(f"{n}:{start}->{end}")
        move(n-1, rest, end)
        return
    else:
        print(f"1:{start}->{end}")
        return


move(num, a, c)

```



代码运行截图

![image-20240312191007321](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312191007321.png)



**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253



【王铭健，工学院，2024年春】用时：20min

思路：

运用递归写法。首先运用列表的拼接将编号为p的人置换至第一位（即列表第一个元素是编号p），然后开始操作：首先取m对现有人数的模r（若为0，则给r赋值为现有人数），然后输出列表第r-1个元素，即为出去的人；再运用列表拼接将第r个元素置换至第一位（这里需要小心数组越界，分类讨论），将现有人数 - 1，再进行以上操作，直至只剩下一个人，然后将这个人对应的编号输出。将这个思路用递归函数写出来即可。

另外注意输入和输出的格式。

##### 代码

```python
# 王铭健，工学院 2300011118
while True:
    n, p, m = map(int, input().split())
    if n == 0 and p == 0 and m == 0:
        break
    queue = [i for i in range(1, n+1)]
    queue = queue[p-1:] + queue[0:p-1]


    def pick(num, data):
        if num > 1:
            No = m % num
            if No == 0:
                No = num
            print(data[No - 1], end=",")
            if No == num:
                data = data[0:No-1]
            else:
                data = data[No:] + data[0:No-1]
            return pick(num-1, data)
        else:
            print(data[0])


    pick(n, queue)

```



代码运行截图

![image-20240312194044190](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312194044190.png)



**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



【王铭健，工学院，2024年春】用时：20min

思路：

显然时间短的应当在前。运用enumerate函数从1开始将每个时间编号，然后根据时间的从小到大将元组（编号，时间）排序，逐个输出编号再输出以对应权值（它后面的人数）作出的加权平均值即可。

##### 代码

```python
# 王铭健，工学院 2300011118
n = int(input())
times = list(map(int, input().split()))
data = list(enumerate(times, start=1))
data.sort(key=lambda x: x[1])
time_sum = 0
for i in range(n):
    time_sum += (n-1-i) * data[i][1]
    print(data[i][0], end=" ")
print()
result = time_sum / n
print(f"{result:.2f}")

```



代码运行截图

![image-20240312201143834](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312201143834.png)



**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



【王铭健，工学院，2024年春】用时：25min

思路：

上学期12月月考题目。

按照题目要求一步一步来即可。

这道题的第一个难点在于如何转化给定的坐标(x, y)为我们想要的x，y两个数的形式。解决方法也很简单，只需要我们先split()把这些坐标逐个分解，然后再取[1:-1]来去掉括号，最后split(",")就可以分解为两个数了。

第二个难点在于对中位数的分析，由于n为奇和偶的时候对中位数的计算不同，所以在我们判断性价比和价格与中位数的大小时，整体需要分奇偶讨论，在每一个情况下分别求出性价比和价格的中位数然后遍历所有数据比较即可。

##### 代码

```python
# 王铭健，工学院 2300011118
n = int(input())
loc = input().split()
prices = list(map(int, input().split()))
dist = []
count = 0
for i in range(n):
    x, y = loc[i][1:-1].split(",")
    dist.append(int(x) + int(y))
divisions = []
for j in range(n):
    divisions.append(dist[j] / prices[j])
divisions_sorted = sorted(divisions)
prices_sorted = sorted(prices)
if n % 2 == 0:
    d_medium = (divisions_sorted[n//2 - 1] + divisions_sorted[n//2]) / 2
    p_medium = (prices_sorted[n//2 - 1] + prices_sorted[n//2]) / 2
    for k in range(n):
        if divisions[k] > d_medium and prices[k] < p_medium:
            count += 1
else:
    d_medium = divisions_sorted[(n - 1) // 2]
    p_medium = prices_sorted[(n - 1) // 2]
    for k in range(n):
        if divisions[k] > d_medium and prices[k] < p_medium:
            count += 1
print(count)

```



代码运行截图

![image-20240312172613221](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312172613221.png)



**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



【王铭健，工学院，2024年春】用时：30min

思路：

上学期期末考试的非必做题。不得不承认，我小看它了...

总体思路上还是简单的，即先将名称与参数量分离，按照规则将参数量转化为可比较的统一大小（我这里用的是：M——1，B——1000），把相同名称的参数量存到字典的同一个键值对里，键为名称，值为参数量；先对键进行字典序从小到大排序，再在每一个键对应的值列表中把所有值从小到大排序，再按照规定格式把这些值转化后逐个输出即可。

问题在于有些参数量为整型，有些为浮点型，如果统一考虑的话最后本该是整型的值会保留一位小数点后的“0”，所以在前后两次转化中都需要分开考虑整型和浮点型的情况，使代码略显冗长。

##### 代码

```python
# 王铭健，工学院 2300011118
from collections import defaultdict
models = defaultdict(list)
n = int(input())
for i in range(n):
    name, amount = input().split("-")
    if amount[-1] == "M":
        if "." in amount:
            amount = float(amount[0:-1])
        else:
            amount = int(amount[0:-1])
    else:
        if "." in amount:
            amount = float(amount[0:-1]) * 1000
        else:
            amount = int(amount[0:-1]) * 1000
    models[name].append(amount)
names = sorted(list(models.keys()))
for j in names:
    result = j + ": "
    data = sorted(models[j])
    for k in data:
        if k >= 1000:
            if k % 1000 == 0:
                result += str(int(k / 1000)) + "B, "
            else:
                result += str(k / 1000) + "B, "
        else:
            result += str(k) + "M, "
    print(result[0:-2])

```



代码运行截图

![image-20240312232620498](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312232620498.png)



## 2. 学习总结和收获

通过月考简单复习了一下dp和递归，可以看出递归确实是这次月考想要考察的重点，应当也是这个月所要学习内容的重要知识基础。做的时候有点惊讶于自己居然把很多模板和函数写法遗忘地如此之快......看来还是要经常复习才是。

这周作业做得有点晚，除了复习课上所学之外也做题很少，确实是其他科目的事情比较多，还有比赛和pre之类的...过了这周后会尽力再跟上数算每日选做的步伐。

另外，我在群里常常看见大家用类写代码，但我个人还是不太习惯，也许后面应该逼迫自己多多尝试类的写法，对笔试应该有一些帮助。





