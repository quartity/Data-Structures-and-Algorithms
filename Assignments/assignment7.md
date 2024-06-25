# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

2024 spring, Complied by 王铭健，工学院



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2

Python编程环境：PyCharm 2023.2.1 (Professional Edition)

C/C++编程环境：暂无



## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



【王铭健，工学院，2024年春】用时：2min

思路：

送分题，reverse秒了。

代码

```python
# 王铭健，工学院 2300011118
s = input().split()
s.reverse()
print(" ".join(s))

```



代码运行截图

![image-20240406154816535](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240406154816535.png)



### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



【王铭健，工学院，2024年春】用时：6min

思路：

使用双端队列deque模拟过程即可。

代码

```python
# 王铭健，工学院 2300011118
from collections import deque
m, n = map(int, input().split())
words = input().split()
queue = deque()
memory = 0
count = 0

for word in words:
    if word not in queue:
        count += 1
        queue.append(word)
        if memory < m:
            memory += 1
        else:
            queue.popleft()
print(count)

```



代码运行截图

![image-20240406155820113](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240406155820113.png)



### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



【王铭健，工学院，2024年春】用时：6min

思路：

在普通情况下，只需要判断排好正序的序列的第k个元素和第k+1个元素是否相同，若相同则不可能有满足条件的最小整数，若不相同则最小整数等于第k个元素。

但是有几个特殊情况要考虑一下：1.k = n，此时所求最小整数等于第n个元素。2.k=0，如果序列最小元素为1，则所求最小整数 = -1（因为其最小只允许取1，不满足条件）；如果最小元素>1，则所求最小整数为1。

允许k = 0着实是很有趣，一开始没有仔细看清楚就自信提交，结果光速WA打脸。

代码

```python
# 王铭健，工学院 2300011118
n, k = map(int, input().split())
seq = sorted(list(map(int, input().split())))
if k == n:
    print(seq[-1])
elif k == 0:
    if seq[0] == 1:
        print("-1")
    else:
        print("1")
else:
    if seq[k-1] == seq[k]:
        print("-1")
    else:
        print(seq[k-1])

```



代码运行截图

![image-20240406161227778](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240406161227778.png)



### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



【王铭健，工学院，2024年春】用时：20min

思路：

按要求二分字符串递归建树，再套用二叉树后序遍历的模板即可。

代码

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, char):
        self.char = char
        self.left = None
        self.right = None


def tree_build(s, n):
    if n == 1:
        if s == "0":
            return TreeNode("B")
        else:
            return TreeNode("I")

    if "0" in s and "1" in s:
        root = TreeNode("F")
    elif "0" not in s and "1" in s:
        root = TreeNode("I")
    else:
        root = TreeNode("B")
    root.left = tree_build(s[0:n//2], n//2)
    root.right = tree_build(s[n//2::], n//2)
    return root


def postorder_traversal_recursion(root):
    if root is None:
        return []

    output = []
    output.extend(postorder_traversal_recursion(root.left))
    output.extend(postorder_traversal_recursion(root.right))
    output.append(root.char)
    return "".join(output)


N = int(input())
print(postorder_traversal_recursion(tree_build(input(), 2**N)))

```



代码运行截图

![image-20240406173826161](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240406173826161.png)



### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



【王铭健，工学院，2024年春】用时：20min

思路：

将编号与小组号的对应存入字典1，将小组号与小组最后一个成员（即下一个成员插入在这个成员的后面）的对应存入字典2。入队时，如果字典2的插入编号对应值为空，则append到队尾；如果非空，则insert到对应值的index + 1。然后将对应值设为插入编号。出队时，后如果出队（popleft）的元素就是其所在小组的最后一个成员（即出队元素 = 字典2的对应值），则将字典2的对应值赋为空。每次出队时print出队元素即可。

代码

```python
# 王铭健，工学院 2300011118
from collections import deque, defaultdict
group = {}
idx = defaultdict(str)
t = int(input())
for i in range(t):
    members = list(input().split())
    for mem in members:
        group[mem] = i

queue = deque()
while True:
    op = input()
    if op == "STOP":
        break
    elif op == "DEQUEUE":
        num = queue.popleft()
        print(num)
        if idx[group[num]] == num:
            idx[group[num]] = ""
    else:
        x = op.split()[1]
        if not idx[group[x]]:
            queue.append(x)
        else:
            queue.insert(queue.index(idx[group[x]])+1, x)
        idx[group[x]] = x

```



代码运行截图

![image-20240406145853360](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240406145853360.png)



### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



【王铭健，工学院，2024年春】用时：25min

思路：

总体思路分为三步：1.通过字典建立输入数据的父子关系；2.找到树的根（这里我将父节点和子节点分别用两个列表记录，最后使用集合减法）；3.通过递归实现要求的从小到大遍历。

感觉这种题目用字典写会更简洁，而且不再需要考虑如何将输入的值全部归入TreeNode类并建立父子关系的问题。

代码

```python
# 王铭健，工学院 2300011118
from collections import defaultdict
n = int(input())
tree = defaultdict(list)
parents = []
children = []
for i in range(n):
    t = list(map(int, input().split()))
    parents.append(t[0])
    if len(t) > 1:
        ch = t[1::]
        children.extend(ch)
        tree[t[0]].extend(ch)


def traversal(node):
    seq = sorted(tree[node] + [node])
    for x in seq:
        if x == node:
            print(node)
        else:
            traversal(x)


traversal((set(parents) - set(children)).pop())

```



代码运行截图

![image-20240406153849165](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240406153849165.png)



## 2. 学习总结和收获

这次月考的题目比较简单，自己大概花了1h15min独立AC6。总体来说考题有相当一部分很水，只要理解概念和题目要求就可以快速做出来。但是我看群里也有很多大佬少于1h就AK了，只能说和他们还有不小差距，而且如果后续考试题目难度加大的话这种差距可能还会更加明显。

另外，在这次月考中我第一次尝试用字典来实现树，起码对这道题而言效果还是很不错的，以后有机会就多加尝试。

最近我的数算学习重点可能还是在理解概念和熟悉模板下的各种写法，再结合作业和每日选做的练习。另外这周二的笔试也给我提了个醒，在闭卷情况下，我对各种算法的实现路径和各类概念及其特点可以说记忆得相当模糊，看来等期中季过后自己得在这方面系统性地加以复习巩固。





