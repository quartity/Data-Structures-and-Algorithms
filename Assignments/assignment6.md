# Assignment #6: "树"算：Huffman,BinHeap,BST,AVL,DisjointSet

Updated 2214 GMT+8 March 24, 2024

2024 spring, Complied by 王铭健，工学院



**说明：**

1）这次作业内容不简单，耗时长的话直接参考题解。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2

Python编程环境：PyCharm 2023.2.1 (Professional Edition)

C/C++编程环境：暂无



## 1. 题目

### 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/



【王铭健，工学院，2024年春】用时：20min

思路：

注意到二叉搜索树的中序遍历就是所有节点值的顺序遍历，因此直接将问题转化为通过前中序遍历写后序遍历，再结合模板即可快速解决。

代码

```python
# 王铭健，工学院 2300011118
def build_postorder(preorder, inorder):
    if inorder and preorder:
        root_val = preorder[0]
        root_index = inorder.index(root_val)

        left_inorder = inorder[:root_index]
        right_inorder = inorder[root_index + 1:]

        left_preorder = preorder[1: root_index + 1]
        right_preorder = preorder[root_index + 1:]

        build_postorder(left_preorder, left_inorder)
        build_postorder(right_preorder, right_inorder)
        print(root_val, end=" ")
        return


n = int(input())
pre = list(map(int, input().split()))
build_postorder(pre, sorted(pre))

```



代码运行截图

![image-20240402230452106](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240402230452106.png)



### 05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/



【王铭健，工学院，2024年春】用时：40min

思路：

先建立一个空的二叉搜索树，再将给定数字序列中的每个数字依次根据插入函数插入树中，由此建树。而后再根据二叉树的层次遍历模板实现层次遍历输出即可。

代码

```python
# 王铭健，工学院 2300011118
from collections import deque


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def insert(node, value):
    if node is None:
        return TreeNode(value)
    if value < node.value:
        node.left = insert(node.left, value)
    elif value > node.value:
        node.right = insert(node.right, value)
    return node


def level_order_traversal(root):
    exp = []
    queue = deque()
    queue.append(root)
    while queue:
        node = queue[0]
        exp.append(queue.popleft().value)
        l = node.left
        r = node.right
        if l:
            queue.append(node.left)
        if r:
            queue.append(node.right)
    return exp


numbers = list(map(int, input().split()))
numbers = list(dict.fromkeys(numbers))  # 去重
root = None
for number in numbers:
    root = insert(root, number)
print(" ".join(map(str, level_order_traversal(root))))

```



代码运行截图

![image-20240402233447688](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240402233447688.png)



### 04078: 实现堆结构

http://cs101.openjudge.cn/practice/04078/

练习自己写个BinHeap。当然机考时候，如果遇到这样题目，直接import heapq。手搓栈、队列、堆、AVL等，考试前需要搓个遍。



【王铭健，工学院，2024年春】用时：1h20min

思路：

根据ppt中的构建二叉堆算法手搓了一遍代码，并在一些重要步骤处加上了注解以便后续复习。

代码

```python
# 王铭健，工学院 2300011118
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    # 将元素上移至合适位置
    def perc_up(self, i):
        while i // 2 > 0:
            # 若子>父，则交换父子位置
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    # 插入新元素：将新元素添加至堆底后上移
    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.perc_up(self.currentSize)

    # 将元素下移至合适位置
    def perc_down(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minchild(i)
            # 若子>父，则交换父子位置
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    # 求某位置元素的最小子女
    def minchild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    # 弹出堆顶的最小值：将堆顶元素移出后将堆底元素置于堆顶，再将其下移以保证堆的性质
    def delmin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.perc_down(1)
        return retval

    # 根据列表建堆：只需下移前1/2的列表元素即可生成堆
    def build_heap(self, alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while i > 0:
            self.perc_down(i)
            i = i - 1


n = int(input())
heap = BinHeap()
for k in range(n):
    s = input()
    if s[0] == '1':
        heap.insert(int(s.split()[1]))
    else:
        print(heap.delmin())

```



代码运行截图

![image-20240402224053720](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240402224053720.png)



### 22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/



【王铭健，工学院，2024年春】用时：1h

思路：

这道题的代码主要是参照ppt上的哈夫曼树写法实现的，在一些重要的步骤上加入了自己的注解，便于后续复习。

代码

```python
# 王铭健，工学院 2300011118
import heapq


class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    # 覆写字符的大小比较
    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight


def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    # 弹出堆中的两个最小元素，再将权值合并后作为新元素入堆
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        # 合并后，char 字段默认值是空
        merged = Node(left.weight + right.weight, None)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


# 编码整个哈夫曼树 用字典将字符与编码对应
def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        # 检测此节点是否对应字符，即是不是叶子节点
        if node.left is None and node.right is None:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes


# 字符 -> 编码
def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded


# 编码 -> 字符
def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        # 检测此节点是否对应字符，即是不是叶子节点
        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded


# 读取输入
n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

# 建哈夫曼树及编码
huffman_tree_root = build_huffman_tree(characters)
codes = encode_huffman_tree(huffman_tree_root)

strings = []
while True:
    try:
        line = input()
        strings.append(line)

    except EOFError:
        break

results = []
for string in strings:
    if string[0] in ('0', '1'):
        results.append(huffman_decoding(huffman_tree_root, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)

```



代码运行截图

![image-20240402215553212](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240402215553212.png)



### 晴问9.5: 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359



【王铭健，工学院，2024年春】用时：1h

思路：

根据平衡二叉树的建立算法，先建立插入函数，再根据树形分析、调整树的平衡模式，最后根据先序遍历序列的模板输出即可。

代码

```python
# 王铭健，工学院 2300011118
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1


class AVL:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)

    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        if balance > 1:
            if value < node.left.value:	 # 树形是 LL
                return self._rotate_right(node)
            else:  # 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:  # 树形是 RR
                return self._rotate_left(node)
            else:  # 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z):
        y = z.right
        t2 = y.left
        y.left = z
        z.right = t2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x

    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)


n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))

```



代码运行截图

![image-20240402235608058](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240402235608058.png)



### 02524: 宗教信仰

http://cs101.openjudge.cn/practice/02524/



【王铭健，工学院，2024年春】用时：

思路：



代码

```python
# 

```



代码运行截图





## 2. 学习总结和收获

在复习二叉树和二叉堆的时候，我注意到了几个有用的结论：

1.对完全二叉树/二叉堆所对应的列表，对在列表中处于位置 p 的节点来说，它的左子节点正好处于位置 2p；同理，右子节点处于位置 2p+1。若要找到树中任意节点的父节点，只需使用 Python 的整数除法即可。给定列表中位置 n 处的节点，其父节点的位置就是 n/2。

2.任意一棵二叉树中，所有叶结点在前序、中序和后序周游序列中的相对次序不发生改变。

3.在一棵二叉树中，结点的总数等于叶子结点（度为0的结点）数、度为1的结点数以及度为2的结点数的总和。同时，除了根结点外，每个结点都是另一个结点的子结点，因此：

总结点数 = 叶子结点数 + 度为1的结点数 + 度为2的结点数

树的总度数 = 结点数 - 1 = 2 x 度为2的结点数 + 1 x 度为1的结点数

这个结论也可以推广至一般树。

4.二叉搜索树的中序遍历序列就是它所有节点值的正序遍历序列。



上周末由于在代表院系参加北大杯赛事，所以实在没有很多时间分给数算，作业的最后一题（并查集）暂时没有完成，后面会自己把这块补上，同时也会尽力多做一些上周落下的每日选做。







