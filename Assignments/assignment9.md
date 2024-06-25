# Assignment #9: 图论：遍历，及 树算

Updated 1739 GMT+8 Apr 14, 2024

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

### 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



【王铭健，工学院，2024年春】用时：50min

思路：

思路大致是： 通过dfs序列建树并顺便计算高度 -> 把树转换为二叉树 -> 计算出二叉树的高度

其中我相对遇到困难的一步是把树转换为二叉树，一开始我想定义两个节点类，但后来发现原来可以在一个class中同时加入“left，right，children”这三个特性，这就让用递归实现树的重组变得很简捷了。

另外，题解中提供的思路很妙，但是我在短时间应该是想不到的...

代码

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None
        self.left = None
        self.right = None


dfs = input()
n = len(dfs)
tree_nodes = [TreeNode(i) for i in range(n)]


def tree_build(nodes, s):
    num = 1
    height = 0
    max_h = 0
    pointer = nodes[0]
    for i in range(n):
        if s[i] == "d":
            pointer.children.append(nodes[num])
            nodes[num].parent = pointer
            pointer = nodes[num]
            num += 1
            height += 1
        else:
            pointer = pointer.parent
            max_h = max(max_h, height)
            height -= 1
    return pointer, max_h


def transform(root):
    if not root.children:
        return
    length = len(root.children)
    root.left = root.children[0]
    transform(root.children[0])
    for j in range(1, length):
        root.children[j-1].right = root.children[j]
        transform(root.children[j])


def tree_height(root):
    if root is None:
        return -1
    return max(tree_height(root.left), tree_height(root.right)) + 1


r, h = tree_build(tree_nodes, dfs)
transform(r)
print(f"{h} => {tree_height(r)}")

```



代码运行截图

![image-20240423195950345](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240423195950345.png)



### 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/



【王铭健，工学院，2024年春】用时：40min

思路：

由于出现"."代表子树为空，只需要在出现"."时return，其余时候通过pop(0)正常进行先序序列建树的递归即可。

代码

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def tree_build(seq):
    if not seq:
        return
    value = seq.pop(0)
    if value == ".":
        return
    root = TreeNode(value)
    root.left = tree_build(seq)
    root.right = tree_build(seq)
    return root


def inorder_traversal_recursion(root):
    if root is None:
        return []
    output = []
    output.extend(inorder_traversal_recursion(root.left))
    output.append(root.value)
    output.extend(inorder_traversal_recursion(root.right))
    return "".join(output)


def postorder_traversal_recursion(root):
    if root is None:
        return []
    output = []
    output.extend(postorder_traversal_recursion(root.left))
    output.extend(postorder_traversal_recursion(root.right))
    output.append(root.value)
    return "".join(output)


s = list(input())
r = tree_build(s)
print(inorder_traversal_recursion(r))
print(postorder_traversal_recursion(r))

```



代码运行截图

![image-20240423223906780](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240423223906780.png)



### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



【王铭健，工学院，2024年春】用时：30min

思路：

看到多次求最小值的题目第一时间想到了使用heapq，随后结合群友的讨论学习使用了懒删除，也就是用字典标记删除掉的元素（及次数）而不去真正的删除它，这样可以有效降低时间复杂度；在heap+懒删除的基础上再建一个辅助栈来显示堆猪的顺序便于标记。

代码

```python
# 王铭健，工学院 2300011118
import heapq
from collections import defaultdict

out = defaultdict(int)
heap = []
stack = []

while True:
    try:
        op = input()
    except EOFError:
        break

    if op == "pop":
        if stack:
            out[stack.pop()] += 1
    elif op == "min":
        if stack:
            while True:
                x = heapq.heappop(heap)
                if not out[x]:
                    heapq.heappush(heap, x)
                    print(x)
                    break
                out[x] -= 1
    else:
        y = int(op.split()[1])
        stack.append(y)
        heapq.heappush(heap, y)
        
```



代码运行截图

![image-20240423221334519](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240423221334519.png)



### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



【王铭健，工学院，2024年春】用时：40min

思路：

一个很常规的DFS，感觉上学期写过不少。把所有可能的方向列出来后逐个尝试再回溯，并用dep储存走的步数、用visited空间储存走过的位置，若遍历完全（m*n = dep）则ans + 1即可。

需要说明的是，我避免麻烦直接全用了10x10的visited空间。不知道为什么如果设置visited空间为nxm时无论怎么调整总会RE。

代码

```python
# 王铭健，工学院 2300011118
dx = [-2, -1, 1, 2, 2, 1, -1, -2]
dy = [1, 2, 2, 1, -1, -2, -2, -1]
ans = 0


def dfs(dep: int, x: int, y: int):
    if n * m == dep:
        global ans
        ans += 1
        return

    for r in range(8):
        s = x + dx[r]
        t = y + dy[r]
        if not visited[s][t] and 0 <= s < n and 0 <= t < m:
            visited[s][t] = True
            dfs(dep + 1, s, t)
            visited[s][t] = False


for i in range(int(input())):
    n, m, x0, y0 = map(int, input().split())
    visited = [[False] * 10 for j in range(10)]
    ans = 0
    visited[x0][y0] = True
    dfs(1, x0, y0)
    print(ans)
    
```



代码运行截图

![image-20240423232416642](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240423232416642.png)



### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



【王铭健，工学院，2024年春】用时：

思路：



代码

```python
# 

```



代码运行截图





### 28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/



【王铭健，工学院，2024年春】用时：

思路：



代码

```python
# 

```



代码运行截图





## 2. 学习总结和收获

1.涉及到树的转换，别忘了可以在一个class中同时加入“left，right，children，parent”等等特性，只要使用环境不冲突就行。

2.以后在动态序列求最值的时候可以试试懒删除，也就是用字典标记删除掉的元素（及次数）而不去真正的删除它，这样可以有效降低时间复杂度；具体实现上可以使用defaultdict(int)。

这周依然是很忙碌的一周，具体来说是因为之前期中季欠了很多课程的债，这周在慢慢还...但分给数算的时间总算是多了起来，也做了几道每日选做的题目，五一假期会继续努力的。





