# Importing all the necessary packages. Don't import any additional packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random

# Loading the dataset
digits = datasets.load_digits()

images = np.array(
    list(digits.images)
)  # np.array() converts the list into a numpy array
labels = np.array(list(digits.target))

print(images.shape)
print(labels.shape)

fig = plt.figure(figsize=(10, 15))
for i in range(1, 11):
    plt.subplot(5, 5, i)
    plt.axis("off")
    plt.imshow(images[i - 1], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(labels[i - 1])
plt.show()


def flatten(images):
    """
    This function takes different images which are 2d arrays each and returns a flattened vector for each image.
    For eg: If an image has m rows and n columns the resulting flattened image will be a vector of size m*n.
    
    Inputs:
        -images: A 3d array of dimensions [N_images, width, height]
        
    Output:
        A 2d array of size [N_images, width*height]
        
    """
    flat_images = np.zeros((len(images), images.shape[1] * images.shape[2]))
    # YOUR CODE HERE
    for i in range(0, images.shape[0]):
        counter = 0
        for j in range(0, images.shape[1]):
            for k in range(0, images.shape[2]):
                flat_images[i, counter] = images[i, j, k]
                counter = counter + 1
    return flat_images

def find_euclidean_distance_2vecs(x1, x2):
    '''
    This function takes as the input 2 vectors x1 and x2 and computes the euclidean distance between them.
    
    Inputs:
        - x1: Vector of size n
        - x2: Vector of size n
        
    Output:
        A scalar value dist which gives the euclidean distance between x1 and x2
    '''
    import math
    dist = 0
    euclidean = np.zeros(len(x1))
    for i in range(0,len(x1)):
    	euclidean[i] = (x1[i]-x2[i])**2
    dist = np.sum(euclidean)
    return math.sqrt(dist)


def find_euclidean_distance_1mat1vec(X, x):
    '''
    This function takes as input a matrix containing m vectors in its each row and a vector of size n 
    and computest distance between the vector with each vector in the matrix
    
    Inputs:
        -X : A matrix of shape [m,n] consisting of m different vectors of size n each
        -x : A vector of size n
        
    Output:
        Returns a vector size m containing distance of each vector in X with x
    
    '''
    dist = np.zeros(len(X))
    
    for i in range(np.shape(X)[0]):
    	dist[i] = find_euclidean_distance_2vecs(X[i,:],x)
    	print(dist) 
    return dist

																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																			def find_euclidean_distance(X1, X2):
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				'''
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					Find L2 distance of each row vector in X2 with each row vector in X1
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					Inputs:
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																						-X1 : A matrix containing m1 n dimensional vectors. [m1, n]
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																						-X2 : A matrix containing m2 n dimensional vectors. [m2, n]
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																						
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					Returns a matrix of shape [m2,m1] containing distance between each point of X2 with each point of X1. For
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					instance the first row of the 																																																																																																																																																																																																																																																																																																																																																																																																																																																																																								output will contain distance of the first vector in X2 with all m1 vectors
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					in X1.
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				'''
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				dist = np.zeros((X2.shape[0],X1.shape[0]))
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				for i in range(0,X2.shape[0]):
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																															for j in range(0,X1.shape[0]):
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																						dist[i,j] = find_euclidean_distance_2vecs(X1[j,:],X2[i,:])
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				print(dist)
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				return dist
    
x = np.reshape([1,2,3,2,3,4],(2,3))
y = np.reshape([2,4,6,4,6,8,1,1,1],(3,3))
find_euclidean_distance(x,y)																																																																																																																																																																																																																					
