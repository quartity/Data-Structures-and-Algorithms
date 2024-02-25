# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by 王铭健，工学院



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2

Python编程环境：PyCharm 2023.2.1 (Professional Edition)

C/C++编程环境：暂无



## 1. 题目

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



【王铭健，工学院，2024年春】用时：10min

思路：

用了和当时计概斐波那契数列一样的记忆化搜索方法，即用值-1代替未被计算过的空位，只计算空位处的数值，其他均直接读取dp表中的值。

当然，这道题也可以用lru_cache迅速解决，但我觉得回顾一下记忆化搜索是很好的。

##### 代码

```python
# 王铭健，工学院 2300011118
n = int(input())
dp = [0, 1, 1] + [-1] * (n-2)


def f(n):
    if dp[n] != -1:
        return dp[n]
    else:
        dp[n] = f(n-1) + f(n-2) + f(n-3)
        return dp[n]


print(f(n))
```



代码运行截图

![image-20240220165113233](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240220165113233.png)



### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



【王铭健，工学院，2024年春】用时：10min

思路：

这道题上学期做过了，但我认为自己之前的写法不是很简捷，所以重写了一个。

总体上就是按照题目要求，利用从左往右的指针在给定字符串中一个一个检索“hello”中的五个字母，如果某一个没有或者不在规定的位置就except出来，print “NO”并退出程序。如果五个都没问题就print “YES”。

##### 代码

```python
# 王铭健，工学院 2300011118
s = input()
try:
    st = s.index("h")
except ValueError:
    print("NO")
    exit()
while True:
    try:
        st += 1
        if s[st] == "e":
            break
    except IndexError:
        print("NO")
        exit()
while True:
    try:
        st += 1
        if s[st] == "l":
            break
    except IndexError:
        print("NO")
        exit()
while True:
    try:
        st += 1
        if s[st] == "l":
            break
    except IndexError:
        print("NO")
        exit()
while True:
    try:
        st += 1
        if s[st] == "o":
            break
    except IndexError:
        print("NO")
        exit()
print("YES")

```



代码运行截图

![image-20240221185141615](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240221185141615.png)



### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



【王铭健，工学院，2024年春】用时：5min

思路：

按照题目要求写代码即可。需要注意的是如果进行了元音的删除，就不需要指针i += 1了；而如果是非元音就需要。所以我在这里用了while True而不是for循环，并最后用IndexError跳出循环。

##### 代码

```python
# 王铭健，工学院 2300011118
vowels = {"a", "e", "i", "u", "o", "y"}
word = list(input().lower())
i = 0
while True:
    try:
        if word[i] in vowels:
            del word[i]
        else:
            word[i] = "." + word[i]
            i += 1
    except IndexError:
        break
print("".join(word))

```



代码运行截图

![image-20240225124913085](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240225124913085.png)



### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



【王铭健，工学院，2024年春】用时：15min

思路：

很久不用欧拉筛，有点忘了怎么写了。这十五分钟很大一部分是在优化欧拉筛的写法。

通过欧拉筛建好质数表之后，就判断sum - 某个质数 是否仍在质数表中，输出符合题意的数字即可。

##### 代码

```python
# 王铭健，工学院 2300011118
def primes(n):
    checklist = [0, 0] + [1] * (n-1)
    result = []
    pointer = 2
    while pointer <= n:
        if checklist[pointer] == 0:
            pointer += 1
            continue
        result.append(pointer)
        st = 2 * pointer
        while st <= n:
            checklist[st] = 0
            st += pointer
        pointer += 1
    return result


n = int(input())
primes_list = primes(n)
i = 0
while primes_list[i] <= n // 2:
    if n - primes_list[i] in primes_list:
        print(primes_list[i], n - primes_list[i])
        exit()
    else:
        i += 1


```



代码运行截图

![image-20240221191104139](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240221191104139.png)



### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



【王铭健，工学院，2024年春】用时：8min

思路：

首先将给定的多项式的每一项分离，然后遍历每一项从而去除掉系数为0的项，并将“n^"后的项次数转化为int型单独提出从而便于比较，最后得到这些项次数的最大值，按要求拼接后输出即可。

##### 代码

```python
# 王铭健，工学院 2300011118
items = input().split("+")
i = 0
while True:
    try:
        if items[i][0] == "0":
            del items[i]
        else:
            a, b = items[i].split("n^")
            items[i] = int(b)
            i += 1
    except IndexError:
        break
print("n^" + str(max(items)))

```



代码运行截图

![image-20240225125943173](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240225125943173.png)



### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



【王铭健，工学院，2024年春】用时：20min

思路：

打一个10^5位的初始值为0的表，代表可能的10^5个编号，哪个编号出现一次就将表对应位置的值+1，这样可以得到一个表的最大值。根据最大值进行index查找，每次找到一个最大值的编号时就将这个编号存下来，同时将表的含此编号之前的部分切掉（也就是只留下后面的部分），从而能不重复地进行下一次index查找，同时需要注意在确定下一个编号的位置后需要加上上一个编号才是实际的编号（因为此前的表被切掉了）。

本来不该花这么长时间的，就是因为我在表的切分上犯了糊涂，debug很久才找到问题所在。

##### 代码

```python
# 王铭健，工学院 2300011118
data = map(int, input().split())
table = [0] * 100000
for i in data:
    table[i-1] += 1
num = max(table)
loc = 0
while True:
    try:
        add = table.index(num) + 1
        loc += add
        print(loc, end=" ")
        table = table[add:]
    except ValueError:
        break

```



代码运行截图

![image-20240225134206292](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240225134206292.png)



## 2. 学习总结和收获

从这周起，python新的征程就算是正式开始了。这周作业里面很多题其实都是之前计概做过的，但是处于恢复手感的考虑，我都用了与之前计概不同的解法，看自己之前的解法我才发现自己以前的思路有多么不成熟（笑）。

新的学期要有新的气象，目前的想法是尽量跟上数算每日选做的步伐，同时每周预习一些后面的内容或者刷题看题解来加深对前面内容的理解。希望最后的成果能让自己满意，我也相信自己会在闫老师的课上学到很多有趣有用的知识。





