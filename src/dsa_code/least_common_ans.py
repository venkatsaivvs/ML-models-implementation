# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # Base case: if root is None or matches p or q
        if not root or root == p or root == q:
            return root
        
        # Search left and right subtrees
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        # If both sides found something → root is LCA
        if left and right:
            return root
        
        # Otherwise, return the non-null side
        return left or right



# Bottom level
node6 = TreeNode(6)
node7 = TreeNode(7)
node4 = TreeNode(4)
node0 = TreeNode(0)
node8 = TreeNode(8)

# Level above
node2 = TreeNode(2, left=node7, right=node4)
node5 = TreeNode(5, left=node6, right=node2)
node1 = TreeNode(1, left=node0, right=node8)

# Root
root = TreeNode(3, left=node5, right=node1)


sol = Solution()

# Test 1: LCA of 5 and 1 → should be 3
lca1 = sol.lowestCommonAncestor(root, node5, node1)
print("LCA of 5 and 1:", lca1.val)  # Output: 3

# Test 2: LCA of 5 and 4 → should be 5
lca2 = sol.lowestCommonAncestor(root, node5, node4)
print("LCA of 5 and 4:", lca2.val)  # Output: 5

# Test 3: LCA of 7 and 4 → should be 2
lca3 = sol.lowestCommonAncestor(root, node7, node4)
print("LCA of 7 and 4:", lca3.val)  # Output: 2
