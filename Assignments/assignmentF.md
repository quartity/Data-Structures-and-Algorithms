# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

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

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



【王铭健，工学院，2024年春】用时：30min

思路：

典型BFS。在遍历每一层之前输出queue（注意queue中只存非空元素）的末尾元素的值即可。

代码

```python
# 王铭健，工学院 2300011118
from collections import deque


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def bfs(root):
    queue = deque()
    queue.append(root)

    while queue:
        print(queue[-1].value, end=" ")
        for i in range(len(queue)):
            cur = queue.popleft()
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)


n = int(input())
nodes = [TreeNode(i) for i in range(1, n+1)]
for i in range(n):
    l, r = map(int, input().split())
    if l != -1:
        nodes[i].left = nodes[l-1]
    if r != -1:
        nodes[i].right = nodes[r-1]
bfs(nodes[0])

```



代码运行截图

![image-20240530001320826](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240530001320826.png)



### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



【王铭健，工学院，2024年春】用时：10min

思路：

见代码注释。

代码

```python
# 如果求右边第一个大于它的元素值，将i+1处改为元素值
n = int(input())
data = list(map(int, input().split()))
stack = []

for i in range(n):
    while stack and data[stack[-1]] < data[i]:  # 此时stack[-1]的右边第一个比它大的元素是data[i]
        data[stack.pop()] = i + 1

    stack.append(i)

while stack:  # 将未能找到右边某元素大于它的元素逐个赋值 
    data[stack[-1]] = 0
    stack.pop()

print(*data)

```



代码运行截图

![image-20240601183809324](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240601183809324.png)



### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



【王铭健，工学院，2024年春】用时：30min

思路：

用Kahn算法判断此有向图是否含环即可。

代码

```python
# 王铭健，工学院 2300011118
from collections import deque, defaultdict


def topological_sort(n, graph):
    in_degree = defaultdict(int)
    result = []
    queue = deque()

    for u in range(1, n+1):
        for v in graph[u]:
            in_degree[v] += 1

    for u in range(1, n+1):
        if in_degree[u] == 0:
            queue.append(u)

    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(result) == n:
        return True
    else:
        return False


for i in range(int(input())):
    n, m = map(int, input().split())
    graph = [[] for j in range(n+1)]
    for k in range(m):
        x, y = map(int, input().split())
        graph[x].append(y)

    if topological_sort(n, graph):
        print("No")
    else:
        print("Yes")

```



代码运行截图

![image-20240605083359367](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240605083359367.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



【王铭健，工学院，2024年春】用时：1h30min

思路：

很有趣也很有借鉴价值的题目。可能是太久没有用二分法了，一开始根本没有往二分查找方向去想，参考题解之后才恍然大悟。

有了二分查找的想法之后，这道题的总体思路就相对很清晰了，也就是在确定所求最小值min的范围之后使用二分法逐次尝试能否找出一个符合题意的分割。不难发现，min的初始最小值是每日开销中的最大值（m=n时一定可以取到），初始最大值是所有每日开销的总和（m=1时可以取到），由此确定了min的范围。为找出符合题意的分割，我们定义了check(x)函数，其意义为“能否在最大月度开销小于x的情况下将这n天划分为小于等于m个月”。check函数的代码实现并不复杂，设初始时仅有一个月，由于每个月代表连续的天数，只需要从第一天开始不停累加每天的开销作为本月的开销，直到超过x值时就月数+1，并把超过这天的开销加进新的月开销中。以此类推，在遍历所有天数后如果月数>m则return False（说明无法找到合适划分），否则return True。在之前确定的min的范围内用二分法不断执行check函数，如果为False就取较大的一半范围，反之取较小的一半范围，直到最终确定min的最小值。

代码

```python
# 王铭健，工学院 2300011118
def check(x):
    months, tem = 1, 0
    for i in range(n):
        if tem + cost[i] > x:
            months += 1
            tem = cost[i]
        else:
            tem += cost[i]

    if months > m:
        return False
    else:
        return True


n, m = map(int, input().split())
cost = []
for i in range(n):
    cost.append(int(input()))

top = sum(cost)
bottom = max(cost)
while bottom < top:
    bisect = (top + bottom) // 2
    if check(bisect):
        top = bisect
    else:
        bottom = bisect + 1

print(top)

```



代码运行截图

![image-20240608114947014](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240608114947014.png)



### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



【王铭健，工学院，2024年春】用时：1h

思路：

一个比较常规的Dijkstra算法解决最短带权路径类问题，所有权均为正数。其思路与“走山路”问题思路大致相同，最大的不同的是多了金币参数需要判定。对三元组（路程，现有钱数，地点），用heapq实现带权BFS，对于所有可行的（即从本地点出发的）道路，如果现有钱数>=道路所需钱数就将其入队（heappush），并创建visited空间储存三元组以防止重复遍历。当到达终点时输出retur路程，或无法按要求到达终点return -1即可。

值得注意的是，可以仅对二元组（地点，现有钱数）创建visited空间，但这样必须在出队（heappop）后再打visited标记，才能保证出队的三元组中的路程是（地点，现有钱数）相同的所有路程中的最小值。

代码

```python
# 王铭健，工学院 2300011118
import heapq
from collections import defaultdict

coins = int(input())
n = int(input())
r = int(input())
roads = defaultdict(list)

for i in range(r):
    start, end, length, cost = map(int, input().split())
    start, end = start - 1, end - 1
    roads[start].append((end, length, cost))


def bfs(st, ed, coins_sum):
    queue = [(0, coins_sum, st)]
    visited = set()

    while queue:
        dis, rest, des = heapq.heappop(queue)
        visited.add((des, dis, rest))
        if des == ed:
            return dis

        for next, l, m in roads[des]:
            if rest >= m:
                new_dis = dis + l
                if (next, new_dis, rest-m) not in visited:
                    heapq.heappush(queue, (new_dis, rest - m, next))

    return -1


print(bfs(0, n - 1, coins))

```



代码运行截图

![image-20240608135551588](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240608135551588.png)



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



【王铭健，工学院，2024年春】用时：

思路：



代码

```python
# 

```



代码运行截图





## 2. 学习总结和收获

这周作业还是很综合的，阅读开销那道题让我拾起了很久没用过的二分法，道路那道题使我更熟悉了带权BFS（也就是Dijkstra）的写法。食物链那道题我认为是很有趣的并查集题目，但同时我觉得对我而言有点太难，从复习机考的角度而言性价比不是很高，因此没有投入很多精力完成它。有点遗憾。

转眼间就到最后一次作业了，这学期的时光真是匆匆。想说的话还是挺多的，可惜上机考近在咫尺，后面还有笔试以及五花八门的其他课程的期末考，所以暂时没有精力去写更多对这学期数算学习的总结和感慨了。更多的话，待到大作业中再慢慢说吧。





