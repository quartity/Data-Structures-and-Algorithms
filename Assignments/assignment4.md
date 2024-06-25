# Assignment #4: 排序、栈、队列和树

Updated 0005 GMT+8 March 11, 2024

2024 spring, Complied by 王铭健，工学院



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

Learn about Time complexities, learn the basics of individual Data Structures, learn the basics of Algorithms, and practice Problems.

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2

Python编程环境：PyCharm 2023.2.1 (Professional Edition)

C/C++编程环境：暂无





## 1. 题目

### 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/



【王铭健，工学院，2024年春】用时：10min

思路：

用deque自带的popleft功能轻松解决。按要求分类讨论实现即可。

代码

```python
# 王铭健，工学院 2300011118
from collections import deque
for i in range(int(input())):
    queue = deque()
    for j in range(int(input())):
        a, b = input().split()
        if a == "1":
            queue.append(int(b))
        else:
            if b == "0":
                queue.popleft()
            else:
                queue.pop()
    if queue:
        for k in queue:
            print(k, end=" ")
        print()
    else:
        print("NULL")

```



代码运行截图

![image-20240319162851415](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240319162851415.png)



### 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



【王铭健，工学院，2024年春】用时：20min

思路：

上学期用函数递归写的死去活来，学了栈之后确实简单多了。

归功于栈先进先出的特色，将表达式分割为列表后反向遍历：如果为数字（注意是浮点数），入栈；如果为运算符，那就弹出栈的最后两个元素，并按照运算符对应规则运算再作为新的元素入栈。最后按规定格式输出栈中仅剩的一个元素即可。

代码

```python
# 王铭健，工学院 2300011118
exp = input().split()
stack = []
for i in range(len(exp)):
    pre = exp[-i-1]
    if pre in "+-*/":
        if pre == "+":
            stack.append(stack.pop() + stack.pop())
        if pre == "-":
            stack.append(stack.pop() - stack.pop())
        if pre == "*":
            stack.append(stack.pop() * stack.pop())
        if pre == "/":
            stack.append(stack.pop() / stack.pop())
    else:
        stack.append(float(pre))
print(f"{stack[0]:.6f}")

```



代码运行截图

![image-20240319161315305](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240319161315305.png)



### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



【王铭健，工学院，2024年春】用时：1h20min

思路：

花了比较长的时间再度理解Shunting Yard算法，不禁再次感叹于它的精妙。

具体算法实现步骤可见week3课件，P21。

代码

```python
# 王铭健，工学院 2300011118
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    stack = []
    postfix = []
    number = ''

    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)
                number = ''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char] <= precedence[stack[-1]]:
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()

    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(str(x) for x in postfix)


for i in range(int(input())):
    exp = input()
    print(infix_to_postfix(exp))

```



代码运行截图

![image-20240319180601157](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240319180601157.png)



### 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/



【王铭健，工学院，2024年春】用时：40min

思路：

首先通过长度和元素是否重复粗判给定序列是否可能为合法序列。然后通过搭建栈来模拟给定序列对应的入栈和出栈过程：如果序列中某一元素 j 在原序列中的index比前一元素的index靠后，那就将这两个index之间的所有原序列元素入栈（第一个元素同理）；反之，从栈顶弹出一个元素并判断它与现在的 j 是否检定一致，如果不一致那显然此序列不合法。按序检索完给定序列中所有元素，若所有检定均一致那么序列合法。

另外，由于输入“若干行”，应使用while True 和 EOFError 异常来终止输入。

代码

```python
# 王铭健，工学院 2300011118
x = input()
while True:
    try:
        possible = True
        stack = []
        seq = input().strip()
        if len(seq) != len(x) or set(seq) != set(x):
            print("NO")
            continue
        num = x.index(seq[0])
        for i in range(num):
            stack.append(x[i])
        for j in range(1, len(x)):
            ind = x.index(seq[j])
            if ind > num:
                for k in range(num+1, ind):
                    stack.append(x[k])
                num = ind
            else:
                top = stack.pop()
                if top != seq[j]:
                    possible = False
                    break
        if possible:
            print("YES")
        else:
            print("NO")
    except EOFError:
        break

```



代码运行截图

![image-20240319202228330](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240319202228330.png)



### 06646: 二叉树的深度

http://cs101.openjudge.cn/practice/06646/



【王铭健，工学院，2024年春】用时：40min

思路：

看到这题第一想法就是用dfs，果然人还是喜欢用自己熟悉的方法。

dfs的思路还是很简单的，从顶点开始，在每个节点处沿left和right两条路径进行深度优先搜索，每进行一次节点跃迁就count += 1，最后得到一个count的最大值即为所求的深度。

代码

```python
# 王铭健，工学院 2300011118
ans, left, right = 1, [-1], [-1]


def dfs(n, count):
    global ans, left, right
    if left[n] != -1:
        dfs(left[n], count + 1)
    if right[n] != -1:
        dfs(right[n], count + 1)
    ans = max(ans, count)


for i in range(int(input())):
    a, b = map(int, input().split())
    left.append(a)
    right.append(b)
dfs(1, 1)
print(ans)

```



代码运行截图

![image-20240319235028704](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240319235028704.png)



### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



【王铭健，工学院，2024年春】用时：2h

思路：

一开始没有想到利用merge sort算法计次，想brute force解出序列的逆序数，但是n<500000的条件让我望而却步...

看群里讨论想到merge sort之后，只需要在其基础上加上对逆序数的计次功能即可，也就是在每次合并（merge）两个已经排好序的列表left和right时当right的某个元素小于left中的某个元素，必需将它移动多少次才能符合升序，就在计次上加多少，直到最后一次合并完成。此时得到的总次数即为逆序数，也就是所求的交换次数。

代码

```python
# 王铭健，工学院 2300011118
# 对 merge sort算法 计次
def merge_sort(lst):
    if len(lst) <= 1:
        return lst, 0

    middle = len(lst) // 2
    left, inv_left = merge_sort(lst[:middle])
    right, inv_right = merge_sort(lst[middle:])

    merged, inv_merge = merge(left, right)

    return merged, inv_left + inv_right + inv_merge


def merge(left, right):
    merged = []
    inv_count = 0
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            inv_count += len(left) - i

    merged += left[i:]
    merged += right[j:]

    return merged, inv_count


while True:
    n = int(input())
    if n == 0:
        break

    seq = []
    for _ in range(n):
        seq.append(int(input()))

    l, inversions = merge_sort(seq)
    print(inversions)

```



代码运行截图

![image-20240319230619314](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240319230619314.png)



## 2. 学习总结和收获

这次的作业明显变难了...本来想在周末搞定的，最后连带着复习week2-4的内容，还是留到了周二才做完。

虽说确实在做题的过程中感觉对栈和队列等等的理解更深刻了，但一做起陌生的题还是感觉到自己的掌握太过浅薄，尤其是和群内的各位大佬相比。

另外，这周最大的进步是终于抽出时间写数算每日选做了，到现在为止差不多做了三十道题，后面虽然确实无法保证每天都按时完成选做，但我一定会尽力跟上步伐，提升自己的做题速度和代码水平（以及跟上群内大佬的讨论www）。





