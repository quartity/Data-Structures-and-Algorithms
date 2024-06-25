# Assignment #8: 图论：概念、遍历，及 树算

Updated 1919 GMT+8 Apr 8, 2024

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

### 19943: 图的拉普拉斯矩阵

matrices, http://cs101.openjudge.cn/practice/19943/

请定义Vertex类，Graph类，然后实现



【王铭健，工学院，2024年春】用时：30min

思路：

一道很好的熟悉Graph和Vertex类的题目，只要搞明白在图中添加节点和边的逻辑就很容易做出来。

代码

```python
# 王铭健，工学院 2300011118
class Graph:
    def __init__(self):
        self.vertices = {}
        self.nums = 0

    def addVertex(self, key):
        new = Vertex(key)
        self.vertices[key] = new
        self.nums += 1
        return new

    def getVertex(self, key):
        if key in self.vertices:
            return self.vertices.get(key)
        else:
            return None

    def __contains__(self, key):
        return key in self.vertices

    def addEdge(self, f, t, weight=0):
        if f not in self.vertices:
            self.addVertex(f)
        if t not in self.vertices:
            self.addVertex(t)
        self.vertices[f].addNeighbor(self.vertices[t], weight)

    def getVertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, key):
        self.id = key
        self.connections = {}

    def addNeighbor(self, nbr, weight=0):
        self.connections[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connections])

    def getConnections(self):
        return self.connections.keys()

    def getId(self):
        return self.id


def construct_L_matrix(n, graph):
    L_matrix = []
    for vertex in graph:
        row = [0] * n
        connections = vertex.getConnections()
        row[vertex.getId()] = len(connections)
        for nbr in connections:
            row[nbr.getId()] -= 1
        L_matrix.append(row)
    return L_matrix


n, m = map(int, input().split())
graph = Graph()
for i in range(n):
    graph.addVertex(i)
for _ in range(m):
    a, b = map(int, input().split())
    graph.addEdge(a, b)
    graph.addEdge(b, a)
matrix = construct_L_matrix(n, graph)
for row in matrix:
    print(" ".join(map(str, row)))

```



代码运行截图

![image-20240421200304665](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240421200304665.png)



### 18160: 最大连通域面积

matrix/dfs similar, http://cs101.openjudge.cn/practice/18160



【王铭健，工学院，2024年春】用时：30min

思路：

这道题是一个计数类的较为典型的DFS类题目。

具体言之，我们建立一个DFS的递归函数，这个函数的起始位置应当在某个“W”上。声明连通域面积为全局变量（global）后，先将这个位置标记为“.”并让连通域面积+1，这样这个位置就被唯一一次添加到了这个连通域的面积中。然后以这个位置为中心进行八向搜索（即将row，col坐标进行加减以把位置移动到邻近的八个位置），如果邻近的某个位置上也是“W”，则再以这个位置为中心进行八向搜索（即DFS函数的递归调用），直至某个位置邻近没有一个“W”。在这个写法中，递归函数不必写return命令，因为实际上函数的意义在于记录这个连通域的面积，无论是否return都已经达到了我们想要的效果。

根据上述的DFS写法，我们对数据矩阵套一层外侧“保护圈”后进行循环遍历，当找到某个位置为“W”时令连通域面积（count）为0并且执行这个位置的DFS函数，这样这个“W”所在的连通域就被记录并且消除了，下一次再找到“W”就会是另一个独立的连通域，这样可以做到不重不漏。利用max函数迭代得到最大的连通域面积再输出即可。但是这道题目将所有面积存入列表再运用max(list)会WA，样例数据是没有问题的，不知道是为什么。

另外，这道题让我对global命令有了更深的理解，如果想要在函数内部声明和改变全局变量的值，那么一定要在函数内部global这个变量，否则等待你的就是下面这一行稀有错误：

UnboundLocalError: cannot access local variable where it is not associated with a value

代码

```python
# 王铭健，工学院 2300011118
oper = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
result = []
count = 0


def dfs(row, col):
    global count

    count += 1
    land[row][col] = "."
    for i in oper:
        t_row = row + i[0]
        t_col = col + i[1]
        if land[t_row][t_col] == "W":
            dfs(t_row, t_col)


t = int(input())
for i in range(t):
    n, m = map(int, input().split())
    land = [['.'] * (m+2)]
    for j in range(n):
        land.append(['.'] + list(input()) + ['.'])
    land.append(['.'] * (m+2))
    max_area = 0
    for k in range(1, n+1):
        for l in range(1, m+1):
            if land[k][l] == "W":
                count = 0
                dfs(k, l)
                max_area = max(max_area, count)
    result.append(max_area)
for p in result:
    print(p)
    
```



代码运行截图

![image-20240421013150510](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240421013150510.png)



### sy383: 最大权值连通块

https://sunnywhy.com/sfbj/10/3/383



【王铭健，工学院，2024年春】用时：40min

思路：

一道比较典型的DFS类计算权值题目，图中的边可以看作两个节点各为对方的连接点，于是可以据此构建一个二维数组表示n个顶点分别的连接点列表，再根据连接点列表对每个节点进行连通块的权值计算。注意在DFS搜索过程中需要创建节点的visited空间以确保各节点的权值不会重复添加。

代码

```python
# 王铭健，工学院 2300011118
def max_weight(n, weights, edges):
    graph = [[] for _ in range(n)]
    for f, t in edges:
        graph[f].append(t)
        graph[t].append(f)

    visited = [False] * n
    w_max = 0

    def dfs(node):
        visited[node] = True
        total_weight = weights[node]
        for neighbor in graph[node]:
            if not visited[neighbor]:
                total_weight += dfs(neighbor)
        return total_weight

    for i in range(n):
        if not visited[i]:
            w_max = max(w_max, dfs(i))

    return w_max


n, m = map(int, input().split())
weights = list(map(int, input().split()))
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))
print(max_weight(n, weights, edges))

```



代码运行截图

![image-20240421201845210](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240421201845210.png)



### 03441: 4 Values whose Sum is 0

data structure/binary search, http://cs101.openjudge.cn/practice/03441



【王铭健，工学院，2024年春】用时：1h30min

思路：

本来以为是道简单题，没想到优化卡的这么死...最后结合了群友们提供的桶的思路才得以简捷地解决。似乎很久没遇到时间和内存都如此大的题目了。

具体做法是将A，B视为一组，利用字典储存所有a+b的结果和重复的次数，再判断每个 -(c+d) 的结果是否在a+b字典的键列表中，如果在就ans + 这个键对应的字典值（也就是这个a+b的重复次数）。最后输出ans即可。如果再用字典储存c+d的结果似乎会MLE。

另外，我一开始在用传统方法时无论是否利用set去重都一直WA，后来才发现原来在判断前三个数的相反数之和是否在D中时我用了”in“来判断，这样就造成了隐性去重（因为D中可能有重复的此数）；还是两两分组运用字典比较明智，既省时又能有效计算重数。

代码

```python
# 王铭健，工学院 2300011118
from collections import defaultdict
n = int(input())
A, B, C, D = [], [], [], []
for _ in range(n):
    a, b, c, d = map(int, input().split())
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)
ab = defaultdict(int)

ans = 0
for a in A:
    for b in B:
        ab[a+b] += 1
sums = ab.keys()
for c in C:
    for d in D:
        if -c-d in sums:
            ans += ab[-c-d]
print(ans)

```



代码运行截图

![image-20240421190049336](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240421190049336.png)



### 04089: 电话号码

trie, http://cs101.openjudge.cn/practice/04089/

Trie 数据结构可能需要自学下。



【王铭健，工学院，2024年春】用时：1h20min

思路：

做此题时学习了前缀树Trie的写法（定义、插入、搜索）以及参考了题解。利用Trie的插入与搜索功能的确让判断号码是否一致变得比较简洁，即只需判断某个号码是否在其他号码组成的前缀树中。

但这里有一处点睛之笔：nums.sort(reverse=True)，这让判断时只需要将所有号码进行一遍”搜索 + 若一致则添加进Trie“的过程，而不会出现”有两个号码不一致，但先短后长而导致判断有误“（例如，18140在先，181403791在后）的情况。

代码

```python
# 王铭健，工学院 2300011118
class TrieNode:
    def __init__(self):
        self.child = {}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, nums):
        curnode = self.root
        for x in nums:
            if x not in curnode.child:
                curnode.child[x] = TrieNode()
            curnode = curnode.child[x]

    def search(self, num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1


t = int(input())
for _ in range(t):
    n = int(input())
    nums = []
    for _ in range(n):
        nums.append(str(input()))
    nums.sort(reverse=True)
    s = 0
    verdict = 1
    trie = Trie()
    for num in nums:
        if trie.search(num):
            print("NO")
            verdict = 0
            break
        trie.insert(num)
    if verdict == 1:
        print('YES')
        
```



代码运行截图

![image-20240421150940999](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240421150940999.png)



### 04082: 树的镜面映射

http://cs101.openjudge.cn/practice/04082/



【王铭健，工学院，2024年春】用时：50min

思路：

由于最终要求输出镜像后的层次遍历序列，所以不需要建树，只需要将每个节点按次序归入对应的层次，输出时再对每个层次的序列取reverse即可。

归入对应层次的方法：首先设根的层次为1，每往下一层层次+1，每个层次对应的节点以字典键值对的形式储存。对于输入的先序遍历序列，分三种情况：1.若是虚结点，则层次-1，因为虚结点出现说明下一个节点为虚结点前一个节点的兄弟节点；2.若是普通节点且为内部节点（0），则将此节点append进当前层次的节点列表（即字典中对应的值），再层次+1，因为根据先序遍历的原则下一个节点应当处于其下一层次；3.若是普通节点且为叶子节点（1），则将此节点append进当前层次的节点列表（即字典中对应的值），再层次-1，因为这一层已经遍历完全了。

代码

```python
# 王铭健，工学院 2300011118
from collections import defaultdict
n = int(input())
preorder = list(input().split())
levels = defaultdict(list)
level = 1

for node in preorder:
    if node == "$1":
        level -= 1
    else:
        levels[level].append(node[0])
        if node[1] == "0":
            level += 1
        else:
            level -= 1

for i in sorted(levels.keys()):
    nodes = levels[i]
    nodes.reverse()
    for j in nodes:
        print(j, end=" ")
        
```



代码运行截图

![image-20240421184503900](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240421184503900.png)



## 2. 学习总结和收获

1.做拉普拉斯矩阵这道题时，注意到了Vertex类中getConnections函数返回的列表中元素是Vertex类元素而非编号key，如果需要使用其编号需要再进行self.getId( )。

2.通过四数之和这道题发现了”in“的判断语句中隐含的去重逻辑，以后对待这种需要计算重数的题目时需要更加谨慎。

3.做镜面映射这道题时，我尝试了直接输出每个节点编号和合并为字符串再统一输出的办法，得到的结论是：前者明显快一些。看来以后可以多选择直接print，也免去合并所花费的时间了。

期中季总算是在本周落下了帷幕，我也总算是能抽出更多时间给数算了。看到每日选做发现已经有一百多题了，然而自己才做了五十余道，实在是有点少。后面会利用五一假期争取迎头赶上的。



​	
