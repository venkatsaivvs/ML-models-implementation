
import numpy as np


X = np.array([[1,1],[1,2], [10,10], [9,10]])
centroids = np.array([[1,1],[10,10]])
labels = np.array([0,0,1,1])

print(X[labels == labels[2]])

print(labels[2])

print(labels == labels[2])

print(centroids[[0,1,0,1]])

distances = np.array([1,2,3,1.2,2,2])
k_neighbours= distances[:3]
#print([dist for dist, _ in k_neighbours])

for i, tranin_point in enumerate(X):
    print(i, tranin_point) 




"""

# Interview Questions & Answers
print("\n" + "="*60)
print("INTERVIEW QUESTIONS & ANSWERS")
print("="*60)

print("\n1. OPTIMIZATION QUESTIONS:")
print("Q: How can you make this more efficient?")
print("A: Use vectorized operations, avoid loops, precompute distances")
print("Q: What's the time complexity?")
print("A: O(n*k*i) where n=points, k=clusters, i=iterations")
print("Q: How would you handle large datasets?")
print("A: Use mini-batch K-means or sample the data")

print("\n2. INITIALIZATION QUESTIONS:")
print("Q: What if random initialization gives bad results?")
print("A: Run multiple times with different random seeds, pick best result")
print("Q: How would you implement K-means++ initialization?")
print("A: Pick first centroid randomly, then pick subsequent ones far from existing centroids")
print("Q: What happens if you pick the same centroid twice?")
print("A: Use replace=False in random.choice to avoid duplicates")

print("\n3. CONVERGENCE QUESTIONS:")
print("Q: How do you know when to stop?")
print("A: Stop when centroid movement is below threshold or max iterations reached")
print("Q: What if the algorithm doesn't converge?")
print("A: Add max iteration limit and return best result so far")
print("Q: How would you handle empty clusters?")
print("A: Reassign random points to empty clusters or remove them")

print("\n4. PARAMETER QUESTIONS:")
print("Q: How do you choose the right K?")
print("A: Use elbow method, silhouette analysis, or domain knowledge")
print("Q: What if K is unknown?")
print("A: Use hierarchical clustering or try multiple K values")
print("Q: How would you validate your clustering?")
print("A: Use silhouette score, inertia, or cross-validation")

print("\n5. DATA HANDLING QUESTIONS:")
print("Q: How would you handle different data types?")
print("A: Convert categorical to numerical, use appropriate distance metrics")
print("Q: What about missing values?")
print("A: Fill missing values or remove incomplete records")
print("Q: How would you scale/normalize the data?")
print("A: Use StandardScaler or MinMaxScaler before clustering")

print("\n6. IMPLEMENTATION QUESTIONS:")
print("Q: How would you make this object-oriented?")
print("A: Create a KMeans class with fit(), predict(), and transform() methods")
print("Q: How would you add visualization?")
print("A: Use matplotlib to plot clusters with different colors")
print("Q: How would you handle edge cases?")
print("A: Add input validation, handle edge cases, and error handling")

print("\n7. ADVANCED QUESTIONS:")
print("Q: How would you implement mini-batch K-means?")
print("A: Update centroids using mini-batches instead of full dataset")
print("Q: What about online K-means?")
print("A: Update centroids incrementally as new data arrives")
print("Q: How would you handle streaming data?")
print("A: Use sliding window or online learning approaches")

print("\n8. COMPARISON QUESTIONS:")
print("Q: How does this compare to other clustering algorithms?")
print("A: K-means is fast but assumes spherical clusters; DBSCAN handles arbitrary shapes")
print("Q: When would you use K-means vs other methods?")
print("A: Use K-means for spherical clusters, DBSCAN for arbitrary shapes, hierarchical for tree structure")
print("Q: What are the limitations of K-means?")
print("A: Assumes spherical clusters, sensitive to initialization, requires predefined K")

print("\n9. REAL-WORLD QUESTIONS:")
print("Q: How would you apply this to customer segmentation?")
print("A: Group customers by purchase behavior, demographics, or RFM analysis")
print("Q: What about image segmentation?")
print("A: Segment images by color, texture, or spatial proximity")
print("Q: How would you handle real-time clustering?")
print("A: Use incremental K-means or online clustering algorithms")

print("\n10. DEBUGGING QUESTIONS:")
print("Q: What if your clusters are imbalanced?")
print("A: Use balanced initialization or weighted K-means")
print("Q: How would you handle outliers?")
print("A: Remove outliers first or use robust distance metrics")
print("Q: What if the data isn't spherical?")
print("A: Use DBSCAN or try different distance metrics like Manhattan distance")

print("\n✅ Interview Q&A section completed!")
"""