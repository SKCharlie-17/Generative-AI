import numpy as np
from numpy.linalg import norm 
A = np.array([2,1,2,3,2,9])
B = np.array([3,4,2,4,5,9])

#compute cosine 
cosine = np.dot(A,B) / (norm(A) * norm(B))
print("Cosine Simlarity", cosine)
