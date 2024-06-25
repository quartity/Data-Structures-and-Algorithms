# Assignment #2: 编程练习

Updated 0953 GMT+8 Feb 24, 2024

2024 spring, Complied by 王铭健，工学院



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2

Python编程环境：PyCharm 2023.2.1 (Professional Edition)

C/C++编程环境：暂无



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



【王铭健，工学院，2024年春】用时：20min

思路：

与课件上基本一致。出于练习考虑，写了一个完整的分数类四则运算和分数比较。

##### 代码

```python
# 王铭健，工学院 2300011118
def gcd(a, b):
    while a % b != 0:
        old_a = a
        old_b = b
        a = old_b
        b = old_a % old_b
    return b


class Fraction:
    def __init__(self, top, bottom):
        self.num = top
        self.den = bottom

    def __str__(self):
        return str(self.num) + "/" + str(self.den)

    def __add__(self, another):
        common = gcd(self.den, another.den)
        new_den = self.den * another.den // common
        new_num = self.num * another.den // common + another.num * self.den // common
        return Fraction(new_num, new_den)

    def __sub__(self, another):
        return Fraction(self.num, self.den) + Fraction(-another.num, another.den)

    def __mul__(self, another):
        new_den = self.den * another.den
        new_num = self.num * another.num
        common = gcd(new_den, new_num)
        return Fraction(new_num // common, new_den // common)

    def __truediv__(self, another):
        return Fraction(self.num, self.den) * Fraction(another.den, another.num)

    def __eq__(self, other):
        first_num = self.num * other.den
        second_num = other.num * self.den
        return first_num == second_num


p, q, m, n = map(int, input().split())
print(Fraction(p, q) + Fraction(m, n))

```



代码运行截图

![image-20240312171339064](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312171339064.png)



### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



【王铭健，工学院，2024年春】用时：10min

思路：

算出每种货物的价值/重量，并依此从高到低对每种货物排序，最后优先取高的货物即可。记得要保留一位小数和加换行符。

##### 代码

```python
# 王铭健，工学院 2300011118
f = lambda: map(int, input().split())
max = 0
n, w = f()
result_list = []
for i in range(n):
    value, weight = f()
    per = value / weight
    result_list.append((per, weight))
result_list.sort(key = lambda x: x[0], reverse = True)
for j in result_list:
    if j[1] <= w:
        w -= j[1]
        max += j[0] * j[1]
    else:
        max += j[0] * w
        break
print("%.1f\n" % max)

```



代码运行截图

![image-20240307190807077](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240307190807077.png)



### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



【王铭健，工学院，2024年春】用时：15min

思路：

通过字典储存每个技能的时刻和伤害，时刻为键，伤害为值，同一时刻的不同技能伤害运用列表储存（这里我用了defaultdict）。然后从小到大检索每个时刻（键），通过伤害列表排序确定这一时刻最多能打出多少伤害，最后判断击杀怪兽的时刻或者“alive”。

##### 代码

```python
# 王铭健，工学院 2300011118
from collections import defaultdict
n = int(input())
for i in range(n):
    damage = 0
    verdict = 1
    ability_dict = defaultdict(list)
    n, m, b = map(int, input().split())
    for j in range(n):
        t, x = map(int, input().split())
        ability_dict[t].append(x)
    for k in sorted(ability_dict.keys()):
        ability_list = ability_dict[k]
        ability_list.sort(reverse = True)
        if m >= len(ability_list):
            damage += sum(ability_list)
        else:
            damage += sum(ability_list[0:m])
        if b <= damage:
            print(k)
            verdict = 0
            break
    if verdict == 1:
        print("alive")

```



代码运行截图

![image-20240307200927232](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240307200927232.png)



### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B



【王铭健，工学院，2024年春】用时：10min

思路：

欧拉筛筛出范围内的所有质数，再判断给出的数字开平方是否在这些质数中。

这道题至少写过四遍了，可以说想思路几乎没有花什么时间。

##### 代码

```python
# 王铭健，工学院 2300011118
from math import ceil
nums = [0, 0] + [1 for i in range(10 ** 6)]
results = []


def primes(n):
    for i in range(2, n+1):
        if nums[i] == 1:
            k = 2
            while k * i <= n:
                nums[k * i] = 0
                k += 1


primes(10 ** 6)
n = int(input())
data = list(map(int, input().split()))
for j in data:
    if ceil(j ** 0.5) == j ** 0.5:
        if nums[ceil(j ** 0.5)]:
            print("YES")
            continue
    print("NO")
```



代码运行截图

![image-20240307211600595](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240307211600595.png)



### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A



【王铭健，工学院，2024年春】用时：10min

思路：

如果所有数加起来不被x整除，则最长子序列长度为n；若能被x整除，那么在首尾放置指针从两头向中间检索，直到找到一个不被x整除的元素为止（设在正数第i位），此时如果删除这个元素那么剩下的子序列必然不被x整除，所以最长子序列长度就是n - i。如果找不到就输出-1。

##### 代码

```python
# 王铭健，工学院 2300011118
t = int(input())
for i in range(t):
    lens = []
    n, x = map(int, input().split())
    nums = list(map(int, input().split()))
    if sum(nums) % x != 0:
        print(n)
    else:
        verdict = 0
        for j in range(n//2 + 1):
            if nums[j] % x != 0 or nums[-j-1] % x != 0:
                print(n - j - 1)
                verdict = 1
                break
        if verdict == 0:
            print("-1")

```



代码运行截图

![image-20240312153026448](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312153026448.png)



### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



【王铭健，工学院，2024年春】用时：10min

思路：

上学期十二月月考题目。

这道题和之前的T-primes非常类似，重点思路都是利用欧拉筛得到质数表然后逐个比对数字是否在表中。考虑到数字过大（范围为10^8），由于有效成绩都可以表示为质数的平方，因此为降低复杂度我选择了将数字开方后先判断它是不是整数，如果是的话再判断它是不是质数，如果是的话就列为有效成绩；这样只需要得到10^4范围内的质数表即可。输出时注意若成绩为0则需直接输出0而非保留两位小数。

##### 代码

```python
# 王铭健，工学院 2300011118
from math import ceil
nums = [0, 0] + [1 for i in range(10 ** 4)]
results = []


def primes(n):
    for i in range(2, n+1):
        if nums[i] == 1:
            k = 2
            while k * i <= n:
                nums[k * i] = 0
                k += 1


primes(10 ** 4)
m, n = map(int, input().split())
for i in range(m):
    scores = list(map(int, input().split()))
    valid = []
    for j in scores:
        if ceil(j ** 0.5) == j ** 0.5:
            if nums[ceil(j ** 0.5)]:
                valid.append(j)
    if not valid:
        results.append(0)
        continue
    results.append(sum(valid) / len(scores))
for k in results:
    if k != 0:
        print("%.2f" % k)
    else:
        print("0")

```



代码运行截图

![image-20240312153756406](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240312153756406.png)



## 2. 学习总结和收获

这次作业的题目除fraction类之外全部是上学期提高班做过的题目，做起来相当熟悉，而且最终思路与上学期基本一致。可以看出上学期的学习还是颇有成效的。fraction类这道题目我是默写出来的，在做完后复习了一遍课件，感觉又有新的收获。

顺带着复习了一下各类模块，比如collections，math和itertools等。最近做题的一个比较主要的目标就是把之前计概里各种各样的工具和模板重新熟悉起来，这样后面会轻松一些。







