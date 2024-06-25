# Assignment #B: 图论和树算

Updated 1709 GMT+8 Apr 28, 2024

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

### 28170: 算鹰

dfs, http://cs101.openjudge.cn/practice/28170/



【王铭健，工学院，2024年春】用时：20min

思路：

比较常规的简单DFS，用visited空间标记遍历过的位置，将一个四向连通块在一次dfs中全部标记，从而达到连通块计数的目的。

另外，题目语言很抽象，有种网上冲浪的美。

代码

```python
# 王铭健，工学院 2300011118
dx = (0, 0, 1, -1)
dy = (1, -1, 0, 0)
visited = [[0] * 10 for i in range(10)]


def dfs(board, x, y):
    visited[x][y] = 1

    for i in range(4):
        x1 = x + dx[i]
        y1 = y + dy[i]
        if 0 <= x1 <= 9 and 0 <= y1 <= 9 and board[x1][y1] == "." and not visited[x1][y1]:
            dfs(board, x1, y1)


Board = []
for j in range(10):
    Board.append(input())
count = 0
for m in range(10):
    for n in range(10):
        if Board[m][n] == "." and not visited[m][n]:
            count += 1
            dfs(Board, m, n)
print(count)

```



代码运行截图

![image-20240507192321125](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240507192321125.png)



### 02754: 八皇后

dfs, http://cs101.openjudge.cn/practice/02754/



【王铭健，工学院，2024年春】用时：30min

思路：

经典得不能再经典的题目。这应该是至少第五遍做了。但实际做的时候并没有想象中那么顺利。

代码

```python
# 王铭健，工学院 2300011118
sol_list = []


def dfs(n, path="", row=0, columns_selected=[], diag1=set(), diag2=set()):
    if row == n:
        sol_list.append(path)
        return
    for col in range(n):
        if col not in columns_selected and col-row not in diag1 and col+row not in diag2:
            dfs(n, path+str(col+1), row+1, columns_selected+[col], diag1|{col-row}, diag2|{col+row})


dfs(8)
n = int(input())
result = []
for i in range(n):
    b = int(input())
    result.append(sol_list[b - 1])
for j in result:
    print(j)

```



代码运行截图

![image-20240507192635870](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240507192635870.png)



### 03151: Pots

bfs, http://cs101.openjudge.cn/practice/03151/



【王铭健，工学院，2024年春】用时：40min

思路：

由于是“shortest sequence”因而必须使用bfs了。

以三种操作作为bfs的六种方向（注意三种操作均可以以两个壶中的任一个为对象），利用queue储存下一步的各个方向，再用一个列表储存每一步的操作路径（注意题目要求的输出格式）。优化的关键是将两个壶的状态用visited空间储存，防止某些操作后两个壶的状态回到之前某一步的状态。以此得到最短路径或者在全部遍历无果后返回impossible。

代码

```python
# 王铭健，工学院 2300011118
def bfs(A, B, C):
    start = (0, 0)
    visited = set()
    visited.add(start)
    queue = [(start, [])]

    while queue:
        (a, b), ops = queue.pop(0)

        if a == C or b == C:
            return ops

        states = [(A, b), (a, B), (0, b), (a, 0), (min(a + b, A),
                  max(0, a + b - A)), (max(0, a + b - B), min(a + b, B))]

        for state in states:
            if state not in visited:
                visited.add(state)
                queue.append((state, ops + [operation(a, b, state)]))

    return "impossible"


def operation(a, b, next_state):
    if next_state == (A1, b):
        return "FILL(1)"
    elif next_state == (a, B1):
        return "FILL(2)"
    elif next_state == (0, b):
        return "DROP(1)"
    elif next_state == (a, 0):
        return "DROP(2)"
    elif next_state == (min(a + b, A1), max(0, a + b - A1)):
        return "POUR(2,1)"
    else:
        return "POUR(1,2)"


A1, B1, C1 = map(int, input().split())
ans = bfs(A1, B1, C1)

if ans == "impossible":
    print(ans)
else:
    print(len(ans))
    for i in ans:
        print(i)

```



代码运行截图 

![image-20240507200357227](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240507200357227.png)



### 05907: 二叉树的操作

http://cs101.openjudge.cn/practice/05907/



【王铭健，工学院，2024年春】用时：50min

思路：

很传统但行之有效的写法。与普通二叉树类题目不同的一点是在TreeNode类中加入了parent特性，此特性包含两个元素（父节点，左/右标记），这样在交换x和y的位置时就可以很方便地找到x和y各自的父节点以及判定其为左子节点还是右子节点，便于调整left和right指针。至于前驱询问，只需要不断将当前节点设为当前节点的左子节点，直至某个当前节点的左子节点为空，则最左边的节点就是此时的当前节点。

代码

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None


def swap(a, b):
    na = nodes[a]
    nb = nodes[b]
    na.parent, nb.parent = nb.parent, na.parent
    if na.parent[1] == "left":
        na.parent[0].left = na
    else:
        na.parent[0].right = na
    if nb.parent[1] == "left":
        nb.parent[0].left = nb
    else:
        nb.parent[0].right = nb


def search(k):
    current = nodes[k]
    while current.left:
        current = current.left
    return current.value


for i in range(int(input())):
    n, m = map(int, input().split())
    nodes = [TreeNode(i) for i in range(n)]
    for j in range(n):
        x, y, z = map(int, input().split())
        if y != -1:
            nodes[x].left = nodes[y]
            nodes[y].parent = (nodes[x], "left")
        if z != -1:
            nodes[x].right = nodes[z]
            nodes[z].parent = (nodes[x], "right")

    for j in range(m):
        op = list(map(int, input().split()))
        if op[0] == 1:
            swap(op[1], op[2])
        else:
            print(search(op[1]))

```



代码运行截图

![image-20240507233548234](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240507233548234.png)





### 18250: 冰阔落 I

Disjoint set, http://cs101.openjudge.cn/practice/18250/



【王铭健，工学院，2024年春】用时：40min

思路：

比较暴力，说实话我自己都没想到能一遍过。

主要思路就是用一个字典表示可乐原始编号与当前所在杯子编号的对应关系，再用一个列表表示各个杯子中所含可乐的原始编号，当x与y对应的可乐不在同一杯（colas[x] != colas[y]）时，将字典中所有y所在杯子中可乐的对应值全部改为x所在杯子编号，再将列表中y所在杯子的所有元素（即所有可乐编号）extend到x所在杯子并将y所在杯子赋空。最后只需查看列表中哪些杯子不是空集即可知道所求的杯子编号和含可乐的杯子总数。

代码

```python
# 王铭健，工学院 2300011118
while True:
    try:
        n, m = map(int, input().split())
        colas = {}
        cups = [[i] for i in range(n+1)]
        for i in range(1, n+1):
            colas[i] = i
        for j in range(m):
            x, y = map(int, input().split())
            cx = colas[x]
            cy = colas[y]
            if cx == cy:
                print("Yes")
            else:
                print("No")
                for num in cups[cy]:
                    colas[num] = cx
                cups[cx].extend(cups[cy])
                cups[cy] = []

        count = 0
        nums = []
        for t in range(1, n+1):
            if cups[t]:
                count += 1
                nums.append(str(t))
        print(count)
        print(" ".join(nums))
    except EOFError:
        break
 
```



代码运行截图

![image-20240507230224859](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240507230224859.png)



### 05443: 兔子与樱花

http://cs101.openjudge.cn/practice/05443/



【王铭健，工学院，2024年春】用时：

思路：



代码

```python
# 

```



代码运行截图





## 2. 学习总结和收获

这周题目感觉难度略有所下降，可能是因为自己上学期选了计概提高班所以对图搜索相对熟悉的原因吧。但是做题时发现自己对并查集不够熟悉，在做冰阔落那道题目时还是习惯使用字典+列表暴力解决而不是使用并查集的思想，虽然AC了但是时间空间复杂度都很高（时间更是并查集方法的四十余倍），如果数据再苛刻一点可能就过不了了。看来还是应该多复习以前的知识点。

另外，今天做完笔试模拟后有点焦虑，感觉笔试内容既多又杂，范围也不是很明确，不好复习，考试题量也很大。真诚希望在最后的复习阶段有一份较为完整的笔试相关内容的讲义资料Orz





