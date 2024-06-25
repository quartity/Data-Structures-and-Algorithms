# Assignment #5: "树"算：概念、表示、解析、遍历

Updated 2124 GMT+8 March 17, 2024

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

### 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/



【王铭健，工学院，2024年春】用时：40min

思路：

建树，找根，求高，数叶，虽然做题过程不太行云流水，但是解题思路还是比较自然清晰。想到每个节点都是子树的根，利用递归解决高度和叶子计数，那么解题就比较容易了。

需要注意的是一开始需要用列表将n个树节点储存起来，便于下方引用。

代码

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



代码运行截图

![image-20240326222823322](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240326222823322.png)



### 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/



【王铭健，工学院，2024年春】用时：30min

思路：

和课件上基本一致。第一步利用栈将括号嵌套树转变为我们熟悉的邻接表树，第二步运用递归写出前序和后序遍历。

另外，由于遍历需要一个起始的node（在TreeNode类下），所以在建树的时候需要最后return一个node，也就是树根。

代码

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def tree_build(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():
            node = TreeNode(char)
            if stack:
                stack[-1].children.append(node)
        elif char == "(":
            if node:
                stack.append(node)
        elif char == ")":
            node = stack.pop()
    return node


def preorder(node):
    print(node.value, end="")
    for child in node.children:
        preorder(child)
    return


def postorder(node):
    for child in node.children:
        postorder(child)
    print(node.value, end="")
    return


root = tree_build(input())
preorder(root)
print()
postorder(root)

```



代码运行截图

![image-20240324141440013](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240324141440013.png)



### 02775: 文件结构“图”

http://cs101.openjudge.cn/practice/02775/



【王铭健，工学院，2024年春】用时：1h20min

思路：

建立class TreeNode时将dirs与files作为两个独立特性加入，然后通过给定格式运用栈建立解析树。建树时注意对files的排序要求，这里运用了lambda x：x.name来使得files有一个合理的排序依据。之后根据要求定义输出函数，把树分层输出为给定的图格式，代码主体就完成了。最后按给定格式接收输入数据并分组输出即可。

代码

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, name):
        self.name = name
        self.dirs = []
        self.files = []


def parse_tree(lst):
    root = TreeNode("ROOT")
    stack = [root]
    for char in lst:
        if char == "]":
            node = stack.pop()
            node.files.sort(key=lambda x: x.name)
        else:
            node = TreeNode(char)
            if char[0] == "d":
                stack[-1].dirs.append(node)
                stack.append(node)
            else:
                stack[-1].files.append(node)
    return root


def display(node, n):
    print("|     " * n + node.name)
    for dir in node.dirs:
        display(dir, n+1)
    for file in node.files:
        print("|     " * n + file.name)


data = []
while True:
    data.append([])
    while True:
        s = input()
        if s == "*" or s == "#":
            data[-1].append("]")
            break
        data[-1].append(s)
    if s == "#":
        data.pop()
        break

for i in range(len(data)):
    print(f"DATA SET {i+1}:")
    root = parse_tree(data[i])
    display(root, 0)
    print()

```



代码运行截图

![image-20240324155410436](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240324155410436.png)



### 25140: 根据后序表达式建立队列表达式

http://cs101.openjudge.cn/practice/25140/



【王铭健，工学院，2024年春】用时：30min

思路：

通过后续表达式建立解析树，然后根据层次遍历再将得到的表达式列表.reverse()即可得到所求的队列表达式。

代码

```python
# 王铭健，工学院 2300011118
from collections import deque


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def build_tree(s):
    stack = []
    for char in s:
        node = TreeNode(char)
        if char.islower():
            stack.append(node)
        else:
            node.right = stack.pop()
            node.left = stack.pop()
            stack.append(node)
    return stack[0]


def lever_order(root):
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
    exp.reverse()
    return exp


for i in range(int(input())):
    print("".join(lever_order(build_tree(input()))))

```



代码运行截图

![image-20240326201138649](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240326201138649.png)



### 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/practice/24750/



【王铭健，工学院，2024年春】用时：30min

思路：

后序遍历的最后一个元素是树的根节点。而在中序遍历序列中，根节点将左右子树分开。通过这种方法找到左右子树的中序遍历序列，并注意到后序遍历序列的前len（左子树对应序列）个元素为左子树的后序遍历序列。据此递归地处理左右子树来构建整个树。

代码

```python
# 王铭健，工学院 2300011118
def build_tree(inorder, postorder):
    if not inorder or not postorder:
        return []

    root_val = postorder[-1]
    root_index = inorder.index(root_val)

    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]

    left_postorder = postorder[:len(left_inorder)]
    right_postorder = postorder[len(left_inorder):-1]

    root = [root_val]
    root.extend(build_tree(left_inorder, left_postorder))
    root.extend(build_tree(right_inorder, right_postorder))

    return root


print("".join(build_tree(input(), input())))

```



代码运行截图

![image-20240326232305332](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240326232305332.png)



### 22158: 根据二叉树前中序序列建树

http://cs101.openjudge.cn/practice/22158/



【王铭健，工学院，2024年春】用时：40min

思路：

本想用与上题相似的思路绕开树解决，但是很久都没有得到正解...无奈只好先老老实实建树，而后再利用后序遍历算法得到所求的后序序列，总分两步。

代码

```python
# 王铭健，工学院 2300011118
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root

def postorder_traversal(root):
    if root is None:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.value

while True:
    try:
        preorder = input()
        inorder = input()
        root = build_tree(preorder, inorder)
        print(postorder_traversal(root))
    except EOFError:
        break

```



代码运行截图

![image-20240326235521774](C:\Users\86181\AppData\Roaming\Typora\typora-user-images\image-20240326235521774.png)



## 2. 学习总结和收获

有个有意思的地方，建解析树的时候list(root)会因为root的数据类型而报错TypeError，但[root]就不会出现问题；相似地，层次遍历的时候deque(root)会因为root的数据类型而报错TypeError，但先建空的deque()再append(root)就不会出现问题。看来这两种方式内部机制不同，以后值得注意。

另外还有一个值得注意的教训：一定要保证每一部分操作的对象是同一个对象，也就是说要注意储存已经编辑好的类中的对象（树节点等）以便下方引用，千万不能在下方再次TreeNode（x），否则会生成一个value相同的新对象，不属于原来的那棵树。



