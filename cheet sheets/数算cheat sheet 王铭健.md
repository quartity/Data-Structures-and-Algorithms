# CS201 Cheat_Sheet_Edition_3

2024.6.5 compiled by 王铭健，23级工学院 (based on 武昱达‘s CS101 Cheat Sheet)

# **一、语法糖**和常用函数

## 1. Part 1

```python
"""语法糖和常用函数"""
print(bin(9)) #bin函数返回二进制，形式为0b1001
dict.items()#同时调用key和value
print(round(3.123456789,5)# 3.12346
print("{:.2f}".format(3.146)) # 3.15
a,b=b,a # 交换a，b
dict.get(key,default=None) # 其中，my_dict是要操作的字典，key是要查找的键，default是可选参数，表示当指定的键不存在时要返回的默认值
ord() # 字符转ASCII
chr() # ASCII转字符
for index,value in enumerate([a,b,c]): # 每个循环体里把索引和值分别赋给index和value。如第一次循环中index=0,value="a" 
```

## 2.part 2

```python
# 二进制转十进制
binary_str = "1010"
decimal_num = int(binary_str, 2) # 第一个参数是字符串类型的某进制数，第二个参数是他的进制，最终转化为整数
print(decimal_num)  # 输出 10
```

# **二、工具**

## 1. **素数筛**

欧拉筛写法：

```python
# 胡睿诚 23数院 
N=20
primes = []
is_prime = [True]*N
is_prime[0] = False;is_prime[1] = False
for i in range(2,N):
    if is_prime[i]:
        primes.append(i)
    for p in primes: #筛掉每个数的素数倍
        if p*i >= N:
            break
        is_prime[p*i] = False
        if i % p == 0: #这样能保证每个数都被它的最小素因数筛掉！
            break
print(primes)
# [2, 3, 5, 7, 11, 13, 17, 19]
```

## 2. 拓展包

### （1） math

```python
import math
print(math.ceil(1.5)) # 2
print(math.pow(2,3)) # 8.0
print(math.pow(2,2.5)) # 5.656854249492381
print(9999999>math.inf) # False
print(math.isqrt(3)) # 1 开平方向下取整
print(math.sqrt(4)) # 2.0
print(math.log(100,10)) # 2.0  math.log(x,base) 以base为底，x的对数
print(math.comb(5,3)) # 组合数，C53
print(math.factorial(5)) # 5！
```

### （2） lru_cache

```python
# 需要注意的是，使用@lru_cache装饰器时，应注意以下几点：
# 1.被缓存的函数的参数必须是可哈希的，这意味着参数中不能包含可变数据类型，如列表或字典。
# 2.缓存的大小会影响性能，需要根据实际情况来确定合适的大小或者使用默认值。
# 3.由于缓存中存储了计算结果，可能导致内存占用过大，需谨慎使用。
# 4.可以是多参数的。
```

### （3）bisect（二分查找）

```python
import bisect
sorted_list = [1,3,5,7,9] #[(0)1, (1)3, (2)5, (3)7, (4)9]
position = bisect.bisect_left(sorted_list, 6)
print(position)  # 输出：3，因为6应该插入到位置3，才能保持列表的升序顺序

bisect.insort_left(sorted_list, 6)
print(sorted_list)  # 输出：[1, 3, 5, 6, 7, 9]，6被插入到适当的位置以保持升序顺序

sorted_list=(1,3,5,7,7,7,9)
print(bisect.bisect_left(sorted_list,7))
print(bisect.bisect_right(sorted_list,7))
# 输出：3 6
```

### （4）年份calendar包

```python
import calendar
print(calendar.isleap(2020)) # True 判断闰年
```

### （5）heapq 优先队列

```python
import heapq # 优先队列可以实现以log复杂度拿出最小（大）元素
lst=[1,2,3]
heapq.heapify(lst) # 将lst优先队列化
heapq.heappop(lst) # 从队列中弹出树顶元素（默认最小，相反数调转）
heapq.heappush(lst,element) # 把元素压入堆中
```

### （6）Counter包

```python
from collections import Counter 
# O(n)
# 创建一个待统计的列表
data = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
# 使用Counter统计元素出现次数
counter_result = Counter(data) # 返回一个字典类型的东西
# 输出统计结果
print(counter_result) # Counter({'apple': 3, 'banana': 2, 'orange': 1})
print(counter_result["apple"]) # 3
```

### （7）default_dict

defaultdict是Python中collections模块中的一种数据结构，它是一种特殊的字典，可以为字典的值提供默认值。当你使用一个不存在的键访问字典时，defaultdict会自动为该键创建一个默认值，而不会引发KeyError异常。

defaultdict的优势在于它能够简化代码逻辑，特别是在处理字典中的值为可迭代对象的情况下。通过设置一个默认的数据类型，它使得我们不需要在访问字典中不存在的键时手动创建默认值，从而减少了代码的复杂性。

使用defaultdict时，首先需要导入collections模块，然后通过指定一个默认工厂函数来创建一个defaultdict对象。一般来说，这个工厂函数可以是int、list、set等Python的内置数据类型或者自定义函数。

```python
from collections import defaultdict
# 创建一个defaultdict，值的默认工厂函数为int，表示默认值为0
char_count = defaultdict(int)
# 统计字符出现次数
input_string = "hello"
for char in input_string:
    char_count[char] += 1
print(char_count)  # 输出 defaultdict(<class 'int'>, {'h': 1, 'e': 1, 'l': 2, 'o': 1})）
```

## (8) itertools包 (排列组合等)

```python
import itertools
my_list = ['a', 'b', 'c']
permutation_list1 = list(itertools.permutations(my_list))
permutation_list2 = list(itertools.permutations(my_list, 2))
combination_list = list(itertools.combinations(my_list, 2))
bit_combinations = list(itertools.product([0, 1], repeat=4)) #repeat=n：在笛卡尔积的所有结果中任取n个合并

print(permutation_list1)
# [('a', 'b', 'c'), ('a', 'c', 'b'), ('b', 'a', 'c'), ('b', 'c', 'a'), ('c', 'a', 'b'), ('c', 'b', 'a')]
print(permutation_list2)
# [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
print(combination_list)
# [('a', 'b'), ('a', 'c'), ('b', 'c')]
print(bit_combinations)
# [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
```



# 三、栈与队列相关

## 1. 单调栈

```python
# 如果求左边第一个大于它的元素索引，反序遍历即可
n = int(input())
data = list(map(int, input().split()))
stack = []

# 写法一
# 如果求右边第一个大于它的元素值，将i+1处改为元素值
for i in range(n):
    while stack and data[stack[-1]] < data[i]:  # 此时stack[-1]的右边第一个比它大的元素是data[i]
        data[stack.pop()] = i + 1  # 此处求的是编号所以需要+1

    stack.append(i)

while stack:  # 将未能找到右边某元素大于它的元素逐个赋值
    data[stack[-1]] = 0
    stack.pop()

print(*data)

# 写法二
result = [0 for i in range(n)]  # 设置找不到右侧比它大的元素时返回的值
for i in range(n-1, -1, -1):
    while stack and data[stack[-1]] <= data[i]:  # 以data[i]为基准，将其右侧不符合要求的全部去除
        stack.pop()

    if stack:
        result[i] = stack[-1] + 1

    stack.append(i)
print(*result)
```

## 2. Shunting Yard算法 (调度场算法)

Shunting Yard 算法的主要思想是使用两个栈（运算符栈和输出栈）来处理表达式的符号。算法按照运算符
的优先级和结合性，将符号逐个处理并放置到正确的位置。最终，输出栈中的元素就是转换后的后缀表达式。
以下是 Shunting Yard 算法的基本步骤：
1. 初始化运算符栈和输出栈为空。
2. 从左到右遍历中缀表达式的每个符号。
    如果是操作数（数字），则将其添加到输出栈。
    如果是左括号，则将其推入运算符栈。
    如果是运算符：
        如果运算符的优先级大于运算符栈顶的运算符，或者运算符栈顶是左括号，则将当前运算符推入运算符栈。
        否则，将运算符栈顶的运算符弹出并添加到输出栈中，直到满足上述条件（或者运算符栈为空）。
    如果是右括号，则将运算符栈顶的运算符弹出并添加到输出栈中，直到遇到左括号。将左括号弹出但不添加到输出栈中。
3. 如果还有剩余的运算符在运算符栈中，将它们依次弹出并添加到输出栈中。
4. 输出栈中的元素就是转换后的后缀表达式。

```
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

## 3. 懒删除

OJ22067 快速堆猪

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

## 4.括号配对/嵌套问题

```python
def check_brackets(s):
    stack = []
    nested = False
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs.keys():
            if not stack or stack.pop() != pairs[ch]:
                return "ERROR"
            if stack:
                nested = True
    if stack:
        return "ERROR"
    return "YES" if nested else "NO"

s = input()
print(check_brackets(s))
```

## 5.双端队列解决回文数字问题

（OJ04067）

```python
from collections import deque

def is_palindrome(num):
    num_str = str(num)
    num_deque = deque(num_str)
    while len(num_deque) > 1:
        if num_deque.popleft() != num_deque.pop():
            return "NO"
    return "YES"

while True:
    try:
        num = int(input())
        print(is_palindrome(num))
    except EOFError:
        break
```



# 四、树相关

## 1.二叉树

### 1.1 二叉树的高度和叶子数目

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


n = int(input())
check = [0] * n
nodes = [TreeNode(i) for i in range(n)]
for i in range(n):
    a, b = map(int, input().split())
    if a != -1:
        check[a] = 1
        nodes[i].left = nodes[a]
    if b != -1:
        check[b] = 1
        nodes[i].right = nodes[b]


def height(root):
    if root is None:
        return -1
    return max(height(root.left), height(root.right)) + 1


def count_leaves(root):
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return 1
    return count_leaves(root.left) + count_leaves(root.right)


r = nodes[check.index(0)]
print(height(r), count_leaves(r))
```

### 1.2 根据二叉树遍历序列建树

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# 前中序建树
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root


# 中后序建树
def build_tree(inorder, postorder):
    if inorder:
        root = TreeNode(postorder[-1])
        root_index = inorder.index(root.value)
        root.right = build_tree(inorder[root_index+1:], postorder[root_index:-1])
        root.left = build_tree(inorder[:root_index], postorder[:root_index])
        return root
```

### 1.3 一般树转换为二叉树

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.left = None
        self.right = None


def transform(root):
    if not root.children:
        return
    length = len(root.children)
    root.left = root.children[0]
    transform(root.children[0])
    for j in range(1, length):
        root.children[j-1].right = root.children[j]
        transform(root.children[j])
```

### 1.4 根据二叉树遍历序列写出另一个遍历序列

```python
# 王铭健，工学院 2300011118
# 前中序-->后序
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
        print(root_val, end="")
        return


# 中后序-->前序
def build_preorder(inorder, postorder):
    if inorder and postorder:
        root_val = postorder[-1]
        root_index = inorder.index(root_val)

        left_inorder = inorder[:root_index]
        right_inorder = inorder[root_index + 1:]

        left_postorder = postorder[:root_index]
        right_postorder = postorder[root_index:-1]

        print(root_val, end="")
        build_preorder(left_inorder, left_postorder)
        build_preorder(right_inorder, right_postorder)
        return


# 由二叉搜索树前序遍历写出后序遍历
"""
王昊 光华管理学院。思路：
建树思路：数组第一个元素是根节点，紧跟着是小于根节点值的节点，在根节点左侧，直至遇到大于根节点值的节点，
后续节点都在根节点右侧，按照这个思路递归即可
"""
class Node():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def buildTree(preorder):
    if len(preorder) == 0:
        return None

    node = Node(preorder[0])

    idx = len(preorder)
    for i in range(1, len(preorder)):
        if preorder[i] > preorder[0]:
            idx = i
            break
    node.left = buildTree(preorder[1:idx])
    node.right = buildTree(preorder[idx:])

    return node


def postorder(node):
    if node is None:
        return []
    output = []
    output.extend(postorder(node.left))
    output.extend(postorder(node.right))
    output.append(str(node.val))

    return output


n = int(input())
preorder = list(map(int, input().split()))
print(' '.join(postorder(buildTree(preorder))))

```

### 1.5 建二叉搜索树

二叉搜索树（Binary Search Tree，BST），它是映射的另一种实现。我们感兴趣的不是元素在树中的确切位置，而是如何利用二叉树结构提供高效的搜索。

二叉搜索树依赖于这样一个性质：小于父节点的键都在左子树中，大于父节点的键则都在右子树中。我们称这个性质为二叉搜索性。

二叉搜索树的中序遍历序列即为数组从小到大排序后的有序序列。

```python
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


numbers = list(map(int, input().split()))
numbers = list(dict.fromkeys(numbers))  # 去重
root = None
for number in numbers:
    root = insert(root, number)
```

## 2.并查集

此处给出union by rank和union by size的写法。

```python
class DisjSet:
    def __init__(self, n):
        # Constructor to create and initialize sets of n items
        self.rank = [1] * n
        self.size = [1] * n
        self.parent = [i for i in range(n)]

    # Finds set of given item x
    def find(self, x):

        # Finds the representative of the set that x is an element of
        if self.parent[x] != x:
            # if x is not the parent of itself
            # Then x is not the representative of its set
            self.parent[x] = self.find(self.parent[x])

        # so we recursively call Find on its parent
        # and move i's node directly under the
        # representative of this set

        return self.parent[x]

    # union by rank
    def Union_by_rank(self, x, y):

        # Find current sets of x and y
        xset = self.find(x)
        yset = self.find(y)

        # If they are already in same set
        if xset == yset:
            return

        # Put smaller ranked item under
        # bigger ranked item if ranks are different
        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset

        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset

        # If ranks are same, then move y under x (doesn't matter
        # which one goes where) and increment rank of x's tree
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1

    # union by size
    def Union_by_size(self, i, j):
        # Find the representatives (or the root nodes) for the set that includes i
        irep = self.find(i)

        # And do the same for the set that includes j
        jrep = self.find(j)

        # Elements are in the same set, no need to unite anything.
        if irep == jrep:
            return

        # Get the size of i’s tree
        isize = self.size[irep]

        # Get the size of j’s tree
        jsize = self.size[jrep]

        # If i’s size is less than j’s size
        if isize < jsize:
            # Then move i under j
            self.parent[irep] = jrep

            # Increment j's size by i's size
            self.size[jrep] += self.size[irep]
        # Else if j’s size is less than i’s size
        else:
            # Then move j under i
            self.parent[jrep] = irep

            # Increment i's size by j's size
            self.size[irep] += self.size[jrep]

# Driver code
obj = DisjSet(5)
obj.Union_by_rank(0, 2)
obj.Union_by_rank(4, 2)
obj.Union_by_rank(3, 1)
if obj.find(4) == obj.find(0):
    print('Yes')
else:
    print('No')
if obj.find(1) == obj.find(0):
    print('Yes')
else:
    print('No')

"""
Yes
No
"""

# 如果要求最终的集合个数，可以使用len(set(x for x in range(1, n + 1) if x == parent[x]))
```

## 3.一般树

### 3.1 计算n-父节点树的高度

```python
# find height of N-ary tree in O(n) (Efficient Approach)

# Recur For Ancestors of node and store height of node at last
def fillHeight(p, node, visited, height):
    if p[node] == -1: # If root node
        visited[node] = 1 # mark root node as visited
        return 0

    if visited[node]:
        return height[node]

    visited[node] = 1

    # recur for the parent node
    height[node] = 1 + fillHeight(p, p[node], visited, height)

    # return calculated height for node
    return height[node]


def findHeight(parent, n):
    ma = 0
    visited = [0] * n
    height = [0] * n

    for i in range(n):
        if not visited[i]:
            height[i] = fillHeight(parent, i, visited, height)

        ma = max(ma, height[i])

    return ma


# Driver Code
if __name__ == '__main__':
    parent = [-1, 0, 0, 0, 3, 1, 1, 2]
    n = len(parent)

    print("Height of N-ary Tree =", findHeight(parent, n))

# Output: Height of N-ary Tree = 2
```

### 3.2 树的各类遍历 ★

```python
class TreeNode:
    def __init(self, value):
        self.value = value
        self.children = []


# 前序遍历
# 若为二叉树，记得更改for写法和children名称并添加：    if root is None:
#                                                   return []
def preorder_traversal_recursion(root):
    output = [root.value]
    for child in root.children:
        output.extend(preorder_traversal_recursion(child))
    return "".join(output)


# (二叉树)中序遍历
def inorder_traversal_recursion(root):
    if root is None:
        return []
    output = []
    output.extend(inorder_traversal_recursion(root.left))
    output.append(root.value)
    output.extend(inorder_traversal_recursion(root.right))
    return "".join(output)


# 后序遍历 栈表达式/后序表达式
# 若为二叉树，记得更改for写法和children名称并添加：    if root is None:
#                                                   return []
def postorder_traversal_recursion(root):
    output = []
    for child in root.children:
        output.extend(postorder_traversal_recursion(child))
    output.append(root.value)
    return "".join(output)


# 层次遍历 队列表达式
from collections import deque


# 1.二叉树
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


# 2.一般树
# Represents a node of an n-ary tree
class Node:
    def __init__(self, key):
        self.key = key
        self.child = []


def newNode(key):
    temp = Node(key)
    return temp


# Prints the n-ary tree level wise
def LevelOrderTraversal(root):
    if (root == None):
        return

    # Standard level order traversal using queue
    q = deque()  # Create a queue
    q.append(root)  # Enqueue root
    while (len(q) != 0):

        n = len(q)

        # If this node has children
        while (n > 0): #写两个while语句是为了划分层级

            # Dequeue an item from queue and print it
            p = q[0]
            q.popleft()
            print(p.key, end=' ')

            # Enqueue all children of the dequeued item
            for i in range(len(p.child)):
                q.append(p.child[i])
            n -= 1

        print()  # Print new line between two levels


# Driver program
if __name__ == '__main__':
    ''' Let us create below tree
                10
            / / \ \
            2 34 56 100
        / \     | / | \
        77 88    1 7 8 9
    '''
    root = newNode(10)
    (root.child).append(newNode(2))
    (root.child).append(newNode(34))
    (root.child).append(newNode(56))
    (root.child).append(newNode(100))
    (root.child[0].child).append(newNode(77))
    (root.child[0].child).append(newNode(88))
    (root.child[2].child).append(newNode(1))
    (root.child[3].child).append(newNode(7))
    (root.child[3].child).append(newNode(8))
    (root.child[3].child).append(newNode(9))

    print("Level order traversal Before Mirroring")
    LevelOrderTraversal(root)

"""
Level order traversal Before Mirroring
10 
2 34 56 100 
77 88 1 7 8 9 
"""
```



# 五、图相关

## 1.graph类与vertex类

❏ Graph() 新建一个空图。
❏ addVertex(vert) 向图中添加一个顶点实例。
❏ addEdge(fromVert, toVert) 向图中添加一条有向边，用于连接顶点fromVert和toVert。
❏ addEdge(fromVert, toVert, weight) 向图中添加一条带权重weight的有向边，用于连接顶点fromVert和toVert。
❏ getVertex(vertKey) 在图中找到名为vertKey的顶点。
❏ getVertices() 以列表形式返回图中所有顶点。
❏ in 通过 vertex in graph 这样的语句，在顶点存在时返回True，否则返回False。

```python
import sys
sys.setrecursionlimit(10000000)


class Graph:
    def __init__(self):
        self.vertices = {}  # 字典key为对应编号，value为Vertex类中元素
        self.nums = 0

    def addVertex(self, key, label):  # 添加节点，id为key（编号或位置），附带数据 label
        new = Vertex(key, label)
        self.vertices[key] = new
        self.nums += 1
        return new

    def getVertex(self, key):  # 返回 id 为 key 的节点
        if key in self.vertices:
            return self.vertices.get(key)
        else:
            return None

    def __contains__(self, key):  # 判断 key 节点是否在图中
        return key in self.vertices

    def addEdge(self, f, t, weight=0):  # 添加从节点 id==f 到 id==t 的边 有向
        if f not in self.vertices:
            self.addVertex(f, None)
        if t not in self.vertices:
            self.addVertex(t, None)
        self.vertices[f].addNeighbor(self.vertices[t], weight)

    def getVertices(self):  # 返回所有的节点 key
        return self.vertices.keys()

    def __iter__(self):  # 重新定义iter，允许迭代每一个节点对象
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, key, label=None):  # 缺省颜色为"white“
        self.id = key
        self.label = label
        self.color = "white"
        self.connections = {}  # 注意：这里connections中的keys是Vertex类中的元素！

    def addNeighbor(self, nbr, weight=0):  # 添加到节点 nbr 的边
        self.connections[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connections])

    def setColor(self, color):  # 设置节点颜色标记
        self.color = color

    def getColor(self):  # 返回节点颜色标记
        return self.color

    def getConnections(self):  # 返回节点的所有邻接节点列表，注意返回的是Vertex类中的元素
        return self.connections.keys()

    def getId(self):  # 返回节点的 id
        return self.id

    def getLabel(self):  # 返回节点的附带数据 label
        return self.label
```

## 2.图搜索

### 2.1 栈实现DFS

```python
from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def DFS(self, v):
        visited = set()
        stack = [v]

        while stack:
            current = stack.pop()
            if current not in visited:
                print(current, end=' ')  # 此处为输出路径
                visited.add(current)
                stack.extend(reversed(self.graph[current]))
```

### **2.2 八皇后的回溯算法(DFS)**

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

### 3.3 队列实现BFS

```python
from collections import defaultdict, deque

# Class to represent a graph using adjacency list

class Graph:
    def __init__(self):
        self.adjList = defaultdict(list)
        
# Function to add an edge to the graph
    def addEdge(self, u, v):
        self.adjList[u].append(v)

    # Function to perform Breadth First Search on a graph represented using adjacency list
    def bfs(self, startNode):
        # Create a queue for BFS
        queue = deque()
        visited = set()

        # Mark the current node as visited and enqueue it
        visited.add(startNode)
        queue.append(startNode)

        # Iterate over the queue
        while queue:
            # Dequeue a vertex from queue and print it
            currentNode = queue.popleft()
            print(currentNode, end=" ")

            # Get all adjacent vertices of the dequeued vertex currentNode
            # If an adjacent has not been visited, then mark it visited and enqueue it
            for neighbor in self.adjList[currentNode]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
```

## 3.图算法

### 3.1 Kahn算法

Kahn算法是基于广度优先搜索（BFS）的一种拓扑排序算法。

Kahn算法的基本思想是通过不断地移除图中的入度为0的顶点，并将其添加到拓扑排序的结果中，直到图中所有的顶点都被移除。具体步骤如下：

1. 初始化一个队列，用于存储当前入度为0的顶点。
2. 遍历图中的所有顶点，计算每个顶点的入度，并将入度为0的顶点加入到队列中。
3. 不断地从队列中弹出顶点，并将其加入到拓扑排序的结果中。同时，遍历该顶点的邻居，并将其入度减1。如果某个邻居的入度减为0，则将其加入到队列中。
4. 重复步骤3，直到队列为空。

Kahn算法的时间复杂度为O(V + E)，其中V是顶点数，E是边数。它是一种简单而高效的拓扑排序算法，在有向无环图（DAG）中广泛应用。

同时Kahn算法可用来判断有向图是否有环。如果 `result` 列表的长度等于图中顶点的数量，则拓扑排序成功，返回结果列表 `result`；否则，图中存在环，无法进行拓扑排序。

```python
# 字典表示图
from collections import deque, defaultdict


def topological_sort(graph):
    in_degree = defaultdict(int)
    result = []
    queue = deque()

    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # 将入度为 0 的顶点加入队列
    for u in graph:
        if in_degree[u] == 0:
            queue.append(u)

    # 执行拓扑排序
    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # 检查是否存在环
    if len(result) == len(graph):
        return result
    else:
        return None


# 示例调用代码
graph = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['F'],
    'E': ['F'],
    'F': []
}

sorted_vertices = topological_sort(graph)
if sorted_vertices:
    print("Topological sort order:", sorted_vertices)
else:
    print("The graph contains a cycle.")
```

### 3.2 Kosaraju算法

Kosaraju算法是一种用于在有向图中寻找强连通分量（Strongly Connected Components，SCC）的算法。它基于深度优先搜索（DFS）和图的转置操作。

Kosaraju算法的核心思想就是两次深度优先搜索（DFS）。

1. **第一次DFS**：在第一次DFS中，我们对图进行标准的深度优先搜索，但是在此过程中，我们记录下顶点完成搜索的顺序。这一步的目的是为了找出每个顶点的完成时间（即结束时间）。

2. **反向图**：接下来，我们对原图取反，即将所有的边方向反转，得到反向图。

3. **第二次DFS**：在第二次DFS中，我们按照第一步中记录的顶点完成时间的逆序，对反向图进行DFS。这样，我们将找出反向图中的强连通分量。

Kosaraju算法的关键在于第二次DFS的顺序，它保证了在DFS的过程中，我们能够优先访问到整个图中的强连通分量。因此，Kosaraju算法的时间复杂度为O(V + E)，其中V是顶点数，E是边数。

```python
def dfs1(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph, neighbor, visited, stack)
    stack.append(node)


def dfs2(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph, neighbor, visited, component)


def kosaraju(graph):
    # Step 1: Perform first DFS to get finishing times
    stack = []
    visited = [False] * len(graph)
    for node in range(len(graph)):
        if not visited[node]:
            dfs1(graph, node, visited, stack)

    # Step 2: Transpose the graph
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)

    # Step 3: Perform second DFS on the transposed graph to find SCCs
    visited = [False] * len(graph)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(transposed_graph, node, visited, scc)
            sccs.append(scc)
    return sccs


# Example
graph = [[1], [2, 4], [3, 5], [0, 6], [5], [4], [7], [5, 6]]
sccs = kosaraju(graph)
print("Strongly Connected Components:")
for scc in sccs:
    print(scc)

"""
Strongly Connected Components:
[0, 3, 2, 1]
[6, 7]
[5, 4]

"""
```

### 3.3 Kruskal算法

Kruskal算法是一种用于解决最小生成树（Minimum Spanning Tree，简称MST）问题的贪心算法。给定一个连通的带权无向图，Kruskal算法可以找到一个包含所有顶点的最小生成树，即包含所有顶点且边权重之和最小的树。

以下是Kruskal算法的基本步骤：

1. 将图中的所有边按照权重从小到大进行排序。

2. 初始化一个空的边集，用于存储最小生成树的边。

3. 重复以下步骤，直到边集中的边数等于顶点数减一或者所有边都已经考虑完毕：

   - 选择排序后的边集中权重最小的边。
   - 如果选择的边不会导致形成环路（即加入该边后，两个顶点不在同一个连通分量中），则将该边加入最小生成树的边集中。

4. 返回最小生成树的边集作为结果。

Kruskal算法的核心思想是通过不断选择权重最小的边，并判断是否会形成环路来构建最小生成树。算法开始时，每个顶点都是一个独立的连通分量，随着边的不断加入，不同的连通分量逐渐合并为一个连通分量，直到最终形成最小生成树。

```python
class DisjointSet:
    def __init__(self, num_vertices):
        self.parent = list(range(num_vertices))
        self.rank = [0] * num_vertices

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False
        else:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1
            return True


def kruskal(graph):
    num_vertices = len(graph)
    edges = []

    # 构建边集
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if graph[i][j] != 0:
                edges.append((i, j, graph[i][j]))

    # 按照权重排序
    edges.sort(key=lambda x: x[2])

    # 初始化并查集
    disjoint_set = DisjointSet(num_vertices)

    # 构建最小生成树的边集
    minimum_spanning_tree = []

    for edge in edges:
        u, v, weight = edge
        if disjoint_set.union(u, v):
            minimum_spanning_tree.append((u, v, weight))

    return minimum_spanning_tree
```

### 3.4 Dijkstra算法

Dijkstra算法用于解决单源最短路径问题，即从给定源节点到图中所有其他节点的最短路径。算法的基本思想是通过不断扩展离源节点最近的节点来逐步确定最短路径。具体步骤如下：

- 初始化一个距离数组，用于记录源节点到所有其他节点的最短距离。初始时，源节点的距离为0，其他节点的距离为无穷大。
- 选择一个未访问的节点中距离最小的节点作为当前节点。
- 更新当前节点的邻居节点的距离，如果通过当前节点到达邻居节点的路径比已知最短路径更短，则更新最短路径。
- 标记当前节点为已访问。
- 重复上述步骤，直到所有节点都被访问或者所有节点的最短路径都被确定。

Dijkstra算法的时间复杂度为O(V^2)，其中V是图中的节点数。当使用优先队列（如最小堆）来选择距离最小的节点时，可以将时间复杂度优化到O((V+E)logV)，其中E是图中的边数。

```python
import heapq
import sys

class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.distance = sys.maxsize
        self.pred = None

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def __lt__(self, other):
        return self.distance < other.distance

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        self.numVertices += 1
        return newVertex

    def getVertex(self, n):
        return self.vertList.get(n)

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

def dijkstra(graph, start):
    pq = []
    start.distance = 0
    heapq.heappush(pq, (0, start))
    visited = set()

    while pq:
        currentDist, currentVert = heapq.heappop(pq)    # 当一个顶点的最短路径确定后（也就是这个顶点
                                                        # 从优先队列中被弹出时），它的最短路径不会再改变。
        if currentVert in visited:
            continue
        visited.add(currentVert)

        for nextVert in currentVert.getConnections():
            newDist = currentDist + currentVert.getWeight(nextVert)
            if newDist < nextVert.distance:
                nextVert.distance = newDist
                nextVert.pred = currentVert
                heapq.heappush(pq, (newDist, nextVert))

# 创建图和边
g = Graph()
g.addEdge('A', 'B', 4)
g.addEdge('A', 'C', 2)
g.addEdge('C', 'B', 1)
g.addEdge('B', 'D', 2)
g.addEdge('C', 'D', 5)
g.addEdge('D', 'E', 3)
g.addEdge('E', 'F', 1)
g.addEdge('D', 'F', 6)

# 执行 Dijkstra 算法
print("Shortest Path Tree:")
dijkstra(g, g.getVertex('A'))

# 输出最短路径树的顶点及其距离
for vertex in g.vertList.values():
    print(f"Vertex: {vertex.id}, Distance: {vertex.distance}")

# 输出最短路径到每个顶点
def printPath(vert):
    if vert.pred:
        printPath(vert.pred)
        print(" -> ", end="")
    print(vert.id, end="")

print("\nPaths from Start Vertex 'A':")
for vertex in g.vertList.values():
    print(f"Path to {vertex.id}: ", end="")
    printPath(vertex)
    print(", Distance: ", vertex.distance)

"""
Shortest Path Tree:
Vertex: A, Distance: 0
Vertex: B, Distance: 3
Vertex: C, Distance: 2
Vertex: D, Distance: 5
Vertex: E, Distance: 8
Vertex: F, Distance: 9

Paths from Start Vertex 'A':
Path to A: A, Distance:  0
Path to B: A -> C -> B, Distance:  3
Path to C: A -> C, Distance:  2
Path to D: A -> C -> B -> D, Distance:  5
Path to E: A -> C -> B -> D -> E, Distance:  8
Path to F: A -> C -> B -> D -> E -> F, Distance:  9
"""
```

带权BFS的另一种写法：（走山路 OJ20106）

```python
# 王铭健，工学院 2300011118
from heapq import heappush, heappop
m, n, p = map(int, input().split())
matrix = []
for i in range(m):
    matrix.append(list(input().split()))
start = []
end = []
for j in range(p):
    x1, y1, x2, y2 = map(int, input().split())
    start.append((x1, y1))
    end.append((x2, y2))
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def judge(x, y, in_queue):
    if 0 <= x < m and 0 <= y < n:
        return matrix[x][y] != "#" and not in_queue[x][y]
    return False


def bfs(x1, y1, x2, y2):
    in_queue = [[0] * n for _ in range(m)]
    in_queue[x1][y1] = 1
    queue = [(0, x1, y1)]
    while queue:
        front = heappop(queue)
        cost = front[0]
        xi = front[1]
        yi = front[2]
        in_queue[xi][yi] = 1
        if xi == x2 and yi == y2:
            return cost
        for i in range(4):
            tx = xi + dx[i]
            ty = yi + dy[i]
            if judge(tx, ty, in_queue):
                heappush(queue, (cost+abs(int(matrix[xi][yi]) - int(matrix[tx][ty])), tx, ty))
    return "NO"


result = []
for k in range(p):
    if matrix[start[k][0]][start[k][1]] == '#'\
       or matrix[end[k][0]][end[k][1]] == '#':
        result.append("NO")
    else:
        result.append(bfs(start[k][0], start[k][1], end[k][0], end[k][1]))
for t in result:
    print(t)
```

### 3.5 Prim算法

通常用于稠密图，且生成的MST一定联通。

```python
import sys
import heapq

class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.distance = sys.maxsize
        self.pred = None

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def __lt__(self, other):
        return self.distance < other.distance

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        self.numVertices += 1
        return newVertex

    def getVertex(self, n):
        return self.vertList.get(n)

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)
        self.vertList[t].addNeighbor(self.vertList[f], cost)

def prim(graph, start):
    pq = []
    start.distance = 0
    heapq.heappush(pq, (0, start))
    visited = set()

    while pq:
        currentDist, currentVert = heapq.heappop(pq)
        if currentVert in visited:
            continue
        visited.add(currentVert)

        for nextVert in currentVert.getConnections():
            weight = currentVert.getWeight(nextVert)
            if nextVert not in visited and weight < nextVert.distance:
                nextVert.distance = weight
                nextVert.pred = currentVert
                heapq.heappush(pq, (weight, nextVert))

# 创建图和边
g = Graph()
g.addEdge('A', 'B', 4)
g.addEdge('A', 'C', 3)
g.addEdge('C', 'B', 1)
g.addEdge('C', 'D', 2)
g.addEdge('D', 'B', 5)
g.addEdge('D', 'E', 6)

# 执行 Prim 算法
print("Minimum Spanning Tree:")
prim(g, g.getVertex('A'))

# 输出最小生成树的边
for vertex in g.vertList.values():
    if vertex.pred:
        print(f"{vertex.pred.id} -> {vertex.id} Weight:{vertex.distance}")

"""
Minimum Spanning Tree:
C -> B Weight:1
A -> C Weight:3
C -> D Weight:2
D -> E Weight:6
"""
```

### 3.6 判断无向图是否联通+有无回路

```python
# 王铭健，工学院 2300011118
def is_connected(graph, n):
    visited = [False] * n  # 记录节点是否被访问过
    stack = [0]  # 使用栈来进行DFS
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


# 读取输入
n, m = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

# 判断连通性和回路
connected = is_connected(graph, n)
has_loop = has_cycle(graph, n)
print("connected:yes" if connected else "connected:no")
print("loop:yes" if has_loop else "loop:no")
```

# 六、排序算法

## 1. 插入排序 Insertion Sort

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/insertionsort.png" alt="insertion-sort" style="zoom:50%;" />



插入排序的基本思想是将待排序的序列分为已排序和未排序两部分，每次从未排序部分选择一个元素插入到已排序部分的适当位置，直到所有元素都被插入到已排序部分为止。

```python
def insertion_sort(arr):							
    for i in range(1, len(arr)):
        j = i										
        while arr[j - 1] > arr[j] and j > 0:		
            arr[j - 1], arr[j] = arr[j], arr[j - 1]
arr = [2, 6, 5, 1, 3, 4]
insertion_sort(arr)
print(arr)

# [1, 2, 3, 4, 5, 6]
```



## 2. 冒泡排序 Bubble Sort

In Bubble Sort algorithm: 

- traverse from left and compare adjacent elements and the higher one is placed at right side.  从左到右扫一遍临项比较，大的置于右侧。
- In this way, the largest element is moved to the rightmost end at first.  每次扫完，当前最大的在最右侧。
- This process is then continued to find the second largest and place it and so on until the data is sorted. 

```python
def bubbleSort(arr):
    n = len(arr)
    for i in range(n):	# (*)
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if (swapped == False):
            break

if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    bubbleSort(arr)
    print(' '.join(map(str, arr)))
```

名词解释：一个pass指(*)处取一个i，一个swap指一次临项交换。

Time Complexity:  $ O(n^{2}) $
Auxiliary Space:  $ O(1)$ 

* 好理解
* 不需要辅助空间
* 稳定（每次对于不同键值排序，例如A指标，排完后其他键值的元素相对位置不变）
* in-place：不需要辅助空间。



## 3. 选择排序 Selection Sort

```python
A = [64, 25, 12, 22, 11]
# 一位一位往下找，确保每一趟后，该位及之前的元素有序。
for i in range(len(A)):
    min_idx = i
    for j in range(i + 1, len(A)):
        if A[min_idx] > A[j]:
            min_idx = j
    A[i], A[min_idx] = A[min_idx], A[i]
    
print(' '.join(map(str, A)))
# Output: 11 12 22 25 64 
```

把最小的前移（或把最大的后移）。

Time Complexity:  $ O(n^{2}) $
Auxiliary Space:  $ O(1)$ 

* 好理解
* 简单数据效率不错
* 不保留顺序（不稳定）

* in-place：不需要辅助空间。



## 4. 快速排序 Quick Sort

```python
def quicksort(arr, left, right):
    # 函数的功能就是把数组从left到right排成顺序。
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)

def partition(arr, left, right):
    # 函数的功能是：把数组从left到right依据pivot分成两部分，其中pivot左边小于pivot,右半部分不小于pivot.
    i = left
    j = right - 1
    pivot = arr[right]
    while i <= j:
        # 筛选不合适的arr[i]，即在pivot左边且大于等于pivot
        while i <= right and arr[i] < pivot:
            i += 1
        # 筛选不合适的arr[j],即在pivot右边且小于pivot
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i


arr = [22, 11, 88, 66, 55, 77, 33, 44]
quicksort(arr, 0, len(arr) - 1)
print(arr)

# [11, 22, 33, 44, 55, 66, 77, 88]
```



Time Complexity:

Best Case: $\Omega(N log (N))$

Average Case: $\Theta ( N log (N))$

Worst Case: $O(N^2)$

Auxiliary Space: O(1), if we don’t consider the recursive stack space.

If we consider the recursive stack space then, in the worst case quicksort could make O(N).

* 高效
* 低内存
* pivot选不好会退化成$ O(n^{2}) $
* 不稳定



## 5. 归并排序 Merge Sort

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2

		L = arr[:mid]	# Dividing the array elements
		R = arr[mid:] # Into 2 halves

		mergeSort(L) # Sorting the first half
		mergeSort(R) # Sorting the second half

		i = j = k = 0
		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1

		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1


if __name__ == '__main__':
	arr = [12, 11, 13, 5, 6, 7]
	mergeSort(arr)
	print(' '.join(map(str, arr)))
# Output: 5 6 7 11 12 13
```

Time Complexity: $ O(N log(N)) $

Auxiliary Space: $ O(N)$ 

* 稳定！
* 最坏情况有所保证（上界是nlogn）
* 可并行



* 需要额外辅助空间
* 小数据量不一定优



## 6.希尔排序 Shell Sort

```python
def shellSort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        # i是子数组末尾元素索引,其确定了一个子数组
        for i in range(gap, n):
            # 下面是一个对子数组的插入排序
            temp = arr[i]
            j = i
            # 保证j-gap>=0,也就是下一个子数组元素索引不越界；
            # 我们理应认为子数组的前面一部分是有序的（参考插入排序）
            # 保证arr[j-gap]<=temp,即找到了合适的插入位置，插入，进入下一个子数组。
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

# 示例
arr = [8, 3, 1, 7, 0, 4, 9, 2, 5, 6]
shellSort(arr)
print(arr)
```

 $O(n^2)$

不需要辅助空间。

* 不稳定



## 7. 排序的比较

|        Name         |  Best   |  Average  |   Worst   | Memory | Stable |       Method        |                         Other notes                          |
| :-----------------: | :-----: | :-------: | :-------: | :----: | :----: | :-----------------: | :----------------------------------------------------------: |
| In-place merge sort |    —    |     —     | $nlog^2n$ |   1    |  Yes   |       Merging       | Can be implemented as a stable sort based on stable in-place merging. |
|      Heapsort       | $nlogn$ |  $nlogn$  |  $nlogn$  |   1    |   No   |      Selection      |                                                              |
|     Merge sort      | $nlogn$ |  $nlogn$  |  $nlogn$  |  *n*   |  Yes   |       Merging       | Highly parallelizable (up to *O*(log *n*) using the Three Hungarian's Algorithm) |
|       Timsort       |   *n*   |  $nlogn$  |  $nlogn$  |  *n*   |  Yes   | Insertion & Merging | Makes *n-1* comparisons when the data is already sorted or reverse sorted. |
|      Quicksort      | $nlogn$ |  $nlogn$  |   $n^2$   | $logn$ |   No   |    Partitioning     | Quicksort is usually done in-place with *O*(log *n*) stack space. |
|      Shellsort      | $nlogn$ | $n^{4/3}$ | $n^{3/2}$ |   1    |   No   |      Insertion      |                       Small code size.                       |
|   Insertion sort    |   *n*   |   $n^2$   |   $n^2$   |   1    |  Yes   |      Insertion      | *O*(n + d), in the worst case over sequences that have *d* inversions. |
|     Bubble sort     |   *n*   |   $n^2$   |   $n^2$   |   1    |  Yes   |     Exchanging      |                       Tiny code size.                        |
|   Selection sort    |  $n^2$  |   $n^2$   |   $n^2$   |   1    |   No   |      Selection      | Stable with O(n) extra space, when using linked lists, or when made as a variant of Insertion Sort instead of swapping the two items. |



## 

# 七、杂项

## 1.单向链表

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def delete(self, value):
        if self.head is None:
            return

        if self.head.value == value:
            self.head = self.head.next
        else:
            current = self.head
            while current.next:
                if current.next.value == value:
                    current.next = current.next.next
                    break
                current = current.next

    def display(self):
        current = self.head
        while current:
            print(current.value, end=" ")
            current = current.next
        print()


# 使用示例
linked_list = LinkedList()
linked_list.append(1)
linked_list.append(2)
linked_list.append(3)
linked_list.display()  # 输出：1 2 3
linked_list.delete(2)
linked_list.display()  # 输出：1 3
```

## 2.双向链表

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_before(self, node, new_node):
        if node is None:  # 如果链表为空，将新节点设置为头部和尾部
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = node
            new_node.prev = node.prev
            if node.prev is not None:
                node.prev.next = new_node
            else:  # 如果在头部插入新节点，更新头部指针
                self.head = new_node
            node.prev = new_node

    def display_forward(self):
        current = self.head
        while current is not None:
            print(current.value, end=" ")
            current = current.next
        print()

    def display_backward(self):
        current = self.tail
        while current is not None:
            print(current.value, end=" ")
            current = current.prev
        print()


# 使用示例
linked_list = DoublyLinkedList()

# 创建节点
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)

# 将节点插入链表
linked_list.insert_before(None, node1)  # 在空链表中插入节点1
linked_list.insert_before(node1, node2)  # 在节点1前插入节点2
linked_list.insert_before(node1, node3)  # 在节点1前插入节点3

# 显示链表内容
linked_list.display_forward()  # 输出：3 2 1
linked_list.display_backward()  # 输出：1 2 3

'''
在这个示例中，定义了一个 `Node` 类表示双向链表中的节点。
每个节点都有一个 `value` 值，以及一个指向前一个节点的 `prev` 指针和一个指向后一个节点的 `next` 指针。

`DoublyLinkedList` 类表示双向链表，它具有 `head` 和 `tail` 两个指针，分别指向链表的头部和尾部。
可以使用 `insert_before` 方法在给定节点 `node` 的前面插入新节点 `new_node`。
如果 `node` 为 `None`，表示在空链表中插入新节点，将新节点设置为头部和尾部。
否则，将新节点的 `next` 指针指向 `node`，将新节点的 `prev` 指针指向 `node.prev`，并更新相邻节点的指针，
把新节点插入到链表中。

`display_forward` 方法用于正向遍历并显示链表中的所有节点。它从头部开始，依次打印每个节点的值。

`display_backward` 方法用于反向遍历并显示链表中的所有节点。它从尾部开始，依次打印每个节点的值。

在示例的最后，创建了一个空的 `DoublyLinkedList` 对象，并创建了三个节点 `node1`、`node2` 和 `node3`。
然后，我们按照顺序将这些节点插入到链表中，并调用 `display_forward` 和 `display_backward` 方法来显示链表的内容。
输出结果应为 `3 2 1` 和 `1 2 3`，分别表示正向和反向遍历链表时节点的值。
'''
```

## 3.Fraction类（熟悉类的写法）

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
```

# 八、 逃生指南

## 1. 除法是否使用地板除得到整数？（否则 4/2=2.0）

## 2. 是否有缩进错误？

## 3. 用于调试的print是否删去？

## 4. 非一般情况的边界情况是否考虑？

## 5. 递归中return的位置是否准确？（缩进问题,逻辑问题）

## 6. 贪心是否最优？有无更优解？

## 7. 正难则反（参考 #蒋子轩 23工院# 乌鸦坐飞机）

## 8. 审题是否准确？ 是否漏掉了输出？（参考）







