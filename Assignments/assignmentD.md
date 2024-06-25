# Assignment #D: May月考

Updated 1654 GMT+8 May 8, 2024

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

### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



【王铭健，工学院，2024年春】用时：5min

思路：

以前的作业题。不再赘述。

代码

```python
# 王铭健，工学院 2300011118
L, M = map(int, input().split())
pt_list = []
for i in range(L+1):
    pt_list.append(1)
for j in range(M):
    st_pt, ed_pt = map(int, input().split())
    for k in range(st_pt, ed_pt+1):
        if pt_list[k] == 1:
            pt_list[k] = 0
        else:
            pass
print(sum(pt_list))

```



代码运行截图

![image-20240521193756371](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240521193756371.png)



### 20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/



【王铭健，工学院，2024年春】用时：10min

思路：

int（x，base）函数立大功。第一次知道原来表示二进制的字符串也可以用int转十进制。

代码

```python
# 王铭健，工学院 2300011118
s = input()
result = []
for i in range(len(s)):
    if int(s[0:i+1], 2) % 5 == 0:
        result.append("1")
    else:
        result.append("0")
print("".join(result))

```



代码运行截图

![image-20240521200612987](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240521200612987.png)



### 01258: Agri-Net

http://cs101.openjudge.cn/practice/01258/



【王铭健，工学院，2024年春】用时：40min

思路：

比较标准的Kruskal算法模板。

但是，我一开始没看见是若干组数据输入，无谓debug了很久...

代码

```python
# 王铭健，工学院 2300011118
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1
            return True
        else:
            return False


def kruskal(graph):
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    edges.sort(key=lambda x: x[2])
    disjoint_set = DisjointSet(n)
    weight_sum = 0

    for edge in edges:
        u, v, weight = edge
        if disjoint_set.union(u, v):
            weight_sum += weight
    return weight_sum


while True:
    try:
        n = int(input())
        matrix = []
        for i in range(n):
            matrix.append(list(map(int, input().split())))
        print(kruskal(matrix))
    except EOFError:
        break

```



代码运行截图

![image-20240521204106021](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240521204106021.png)



### 27635: 判断无向图是否连通有无回路(同23163)

http://cs101.openjudge.cn/practice/27635/



【王铭健，工学院，2024年春】用时：40min

思路：

判断是否连通和是否成环均可使用DFS。

值得一提的是，这道题我做完后参考了一下题解的思路，学习到了使用栈进行DFS的操作而不是递归。这也是个不错的方法。

代码

```python
# 王铭健，工学院 2300011118
def is_connected(graph, n):
    visited = [False] * n 
    stack = [0] 
    visited[0] = True

    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                stack.append(neighbor)
                visited[neighbor] = True

    return all(visited)


def has_cycle(graph, n):
    def dfs(node, visited, parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, visited, node):
                    return True
            elif parent != neighbor:
                return True
        return False

    visited = [False] * n
    for node in range(n):
        if not visited[node]:
            if dfs(node, visited, -1):
                return True
    return False


n, m = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)


connected = is_connected(graph, n)
has_loop = has_cycle(graph, n)
print("connected:yes" if connected else "connected:no")
print("loop:yes" if has_loop else "loop:no")

```



代码运行截图

![image-20240521214835518](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240521214835518.png)



### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/



【王铭健，工学院，2024年春】用时：1h20min

思路：

一开始想用bisect插入暴力解决，但是一直TLE，然后无奈从tag里的heap思考算法。后来终于想到用一个大根堆（小根堆元素全取负数来实现）表示比当前中位数小的数字，再用一个小根堆表示比当前中位数大的数字。然后遍历所有数，在保持当前中位数位于大根堆的顶端的前提下将新的数插入大根堆或小根堆。值得注意的是，为了满足这个前提，每一次插入后都要检查大根堆和小根堆的元素个数，使其满足len（大根堆）- len（小根堆）= 0（若当前共遍历了偶数个数）或1（若当前共遍历了奇数个数），否则进行两个堆间的转移操作。在每次遍历个数为奇数时输出一次中位数即可。

代码

```python
# 王铭健，工学院 2300011118
import heapq


def dynamic_median(nums):
    max_heap = []
    min_heap = []

    median = []
    for i, num in enumerate(nums):
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)

        if len(max_heap) - len(min_heap) > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))

        if i % 2 == 0:
            median.append(-max_heap[0])

    return median


t = int(input())
for j in range(t):
    data = list(map(int, input().split()))
    m = dynamic_median(data)
    print(len(m))
    print(*m)

```



代码运行截图

![image-20240521225711694](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240521225711694.png)



### 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/



【王铭健，工学院，2024年春】用时：

思路：



代码

```python
# 

```



代码运行截图





## 2. 学习总结和收获

通过这次月考学习了单调栈的模板和简单应用，并提高了对Kruskal算法和heap应用的熟悉度。但是奶牛排队暂时没时间做了，等过两天写完习概论文会花时间搞定。

值得一提的是，在查看题解后发现了一个小技巧：

print(list) = [1，2，3，4，5]  -> print(*list) = 1 2 3 4 5

以前都是用先把列表元素转换为str再用join函数的，这下找到简洁的替代了。



这周是相当忙碌的一周，各种pre加论文群魔乱舞。等过几天这些事情结束之后就专心致志准备数算的机考了。





