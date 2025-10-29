import numpy as np

x = np.array([[1,2],
             [3,4],
             [5,6]])
labels = np.array([0,1,0])
centroids = np.array([[1,1],[2,2]])
centroids[labels]
print(centroids[labels])



