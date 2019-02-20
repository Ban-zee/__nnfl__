from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

datapoints = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(datapoints.data, datapoints.target, stratify=datapoints.target, random_state=42)
y_train=y_train.reshape(y_train.shape[0],1)
y_test=y_test.reshape(y_test.shape[0],1)



#The function will modify X_train by keeping only the first two features 
def modify(X):
    '''
    Inputs:
        -X : Data matrix containing m data points each having n features. Shape [m, n]
    Outputs:
        Modified data matrix containing just the first 2 features of the data points. Shape [m, 2]
    '''
    
    return(X[:,:2])

print("Running sample test case")
np.random.seed(0)
a = np.random.rand(10, 2)
print(a)
print(modify(a))



#define the sigmoid function (0.5 marks)
def sigmoid(z):
    '''
    Inputs:
        -z : z can be either a number, a numpy 1d array or a numpy 2d array
    Output:
        sigmoid function applied to z, if z is a vector or a matrix the function must be applied to its each element
        
    '''
    return 1/(1+np.exp(-1*z))

print(sigmoid(0))
print("Running sample test case 1:")
assert np.allclose(0.5, sigmoid(0), atol = 1e-5)
print("Test case passed")
print("Running sample test case 2:")
assert np.allclose(np.array([0.73105858, 0.73105858, 0.73105858]), sigmoid(np.array([1,1,1])), atol = 1e-5)
print("Test case passed")
print("Running sample test case 3")
assert np.allclose(np.array([[0.73105858, 0.88079708],
       [0.5       , 0.95257413]]), sigmoid(np.array([[1,2],[0,3]])), atol = 1e-5)
print("Test case passed")

def log_loss(y,yhat): 
    '''
    Inputs:
        -y : The actual labels from the dataset. Shape [m,1]
        -yhat : Model predictions. Shape [m,1]
    
    Output:
        A scalar value representing the log loss as defined above
    '''
    import math
    if not np.isscalar(y):
    	N = len(y)
    	sum = 0
    	for i in range(0,N):
    		sum = sum + y[i]*math.log(yhat[i])+(1-y[i])*math.log(1-yhat[i])
    	return -(sum/N)
    else:
    	sum = y*math.log(yhat)+(1-y)*math.log(1-yhat)
    	return -sum
print(log_loss([1,0.76,0.87],[0.43,0.56,0.99]))
print(log_loss(1,0.5))

def intialize():
    '''
    Return W and b with W being a column vector of size 2 containing 0 and b being a scalar value 0
    '''
    W = np.zeros((2,1))
    b=0
    # YOUR CODE HERE
    print(W)
    print(b)
    return [W,b]
    
'''Implement the update equations for gradient descent and run it for 10000 iterations.
   Append the loss values to losses list after every 100 iterations'''
# (1.5 marks)
def train(X_train,y_train, lr = 0.02):
    '''
    Inputs:
        - X_train : Training data matrix. shape [m,n]
        - y_train : Training labels. shape [m,1]
        - lr : learning rate for the gradient descent algorithm
    Outputs:
        weights W (shape [2,1]) and bias b (scalar value)
    '''
    losses=[]
    X_train = modify(X_train)
    [W,b]=intialize()
    for epoch in range(10000):
        m=len(X_train)
        z = np.dot(X_train,W)+b
        y_pred = sigmoid(z)
        loss = log_loss(y_train,y_pred)
        if epoch%100==0:
        	losses.append(loss)
        diff = y_pred-y_train
        theta_1 = lr*np.sum(diff*np.reshape(X_train[:,0],(m,1)))/m
        theta_2 = lr*np.sum(diff*np.reshape(X_train[:,1],(m,1)))/m
        theta_b = lr*np.sum(diff)/m
        W[0,0] = W[0,0]-theta_1
        W[1,0] = W[1,0]-theta_2
        b = b-theta_b
    return [W,b,losses]
W,b,a = train(X_train,y_train)
print(W)
plt.plot(a)
plt.show()


#calculate the labels for training examples and populate the list preds (0.5 marks)
def predict(X, W, b):
    '''
    Inputs:
        -X : Data matrix. Shape [m,n]
        -W : Weights of logistic regression model
        -b : bias of logistic regression model
    
    Output:
        predictions array of size m containing 0's or 1's representing negative and postive class respectively. 
    '''
    preds=[]
    X = modify(X)
    y_pred =  sigmoid(np.dot(X,W)+b)
    for i in range(0,y_pred.shape[0]):
        if y_pred[i]>=0.5:
            preds.append(1)
        else:
            preds.append(0)
    return preds
    
print("Running sample test case 1")
preds=predict(X_train, W, b)
assert np.allclose(preds[0],0)
print("Test case passed")
print("Running sample test case 2")
assert np.allclose(preds[1],1)
print("Test case passed")


def find_accuracy(y_preds, y_true):
    '''
    Calculates the accuracy of the classifier.
    
    Inputs:
        -y_preds : Predictions by KNN Classifier
        -y_true : Actual labels
        
    Output:
        Accuracy in percentage which is defined as : 100*number_of_correctly_classified_examples/total_examples
    '''
    
    acc = 0
    for i in range(0,len(y_preds)):
        if y_true[i]==y_preds[i]:
            acc = acc+1
    acc = acc/len(y_preds)
    return acc
    
    
preds_train = predict(X_train, W, b)
preds_test = predict(X_test, W, b)
acc_train = find_accuracy(preds_train, y_train)
acc_test = find_accuracy(preds_test, y_test)
print(acc_train)
print(acc_test)

preds_train = predict(X_train, W, b)
preds_test = predict(X_test, W, b)
acc_train = find_accuracy(preds_train, y_train)
acc_test = find_accuracy(preds_test, y_test)
print(acc_train)
print(acc_test)


#the following code prints the decision boundry for our problem on the training dataset
plt.scatter(X_train[:,0],X_train[:,1],c=y_train.ravel())
ax=plt.gca()
xvals=np.array(ax.get_xlim()).reshape(-1,1)
yvals=-(xvals*W[0][0]+b)/W[1][0]
plt.plot(xvals,yvals)
plt.ylim(0,40)
plt.show()
