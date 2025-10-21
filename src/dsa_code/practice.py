import numpy as np
x = np.array([1,2,3])
y = np.array([[1],
             [2],
             [3]])

print(np.dot(x, y))
print(x@y)

a = np.array([[1,2],
            [3,4]])
b = np.array([[3,4],
             [5,6]])

print(np.dot(a, b))
print(a@b)

c = np.array([1,2,3])
d = np.array([4,5,6])
print(np.dot(c, d))
print(c@d)

