import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'heart.csv'

data = pd.read_csv(r'heart.csv')
df = pd.DataFrame(data, columns=['trestbps', 'chol', 'thalach', 'oldpeak', 'target'])
print('df = ')
print(df.head(10))
print('===============================================================')
print(df.describe())

positive = df[df['target'].isin([1])]
negative = df[df['target'].isin([0])]

# print('with heart disease \n', positive)
# print('without heart disease \n', negative)

#fig, ax = plt.subplots(figsize=(8, 5))
#ax.scatter(positive['trestbps'], positive['chol'], positive['thalach'], positive['oldpeak'],
#           s=50, c='b', marker='o', label='with heart disease')

#ax.scatter(negative['trestbps'], negative['chol'], negative['thalach'], negative['oldpeak'],
#          s=50, c='r', marker='x', label='without heart disease')

#sigmoid function plotting
def sigmoid(z):
    return 1 / (1+np.exp(-z))

nums = np.arange(-10,10,step=1)

fig, ax= plt.subplots(figsize=(8,5))
ax.plot(nums,sigmoid(nums), 'r')
#plt.show()

#bnzwd l column l awl lly b 1s
df.insert(0,'ones',1)
print('new data \n', df)

#bfsl l data l x 3n l y
cols= df.shape[1]
X= df.iloc[:,0:cols-1]
Y= df.iloc[:,cols-1:cols]

print('x \n',X)
print('y \n',Y)

#converting to numpy arrays and intialize the parameter array theta
X= np.array(X.values)
Y= np.array(Y.values)
theta = np.zeros(5)
print('theta = n', theta)
print('x \n',X)
print('y \n',Y)

#to print the shape of x,y,theta
print('X.shape =' , X.shape)
print('Y.shape =' , Y.shape)
print('theta.shape =' , theta.shape)

# cost function
def cost(thetav,Xv,Yv):
    thetav = np.matrix(thetav)
    print('thetav = ', thetav)
    Xv = np.matrix(Xv)
    Yv = np.matrix(Yv)
    # first term in the cost function
    first = np.multiply(-Yv,np.log(sigmoid(Xv * thetav.T)))
    #second term in the cost function
    second = np.multiply((1-Yv),np.log(1-sigmoid(Xv * thetav.T)))
    return np.sum(first-second) / (len(Xv))


#using cost function
thiscost= cost(theta,X,Y)
print('cost =' ,thiscost)

alpha = 0.005
#gradient function
def gradient(thetav, Xv,Yv):
    thetav = np.matrix(thetav)
    Xv = np.matrix(Xv)
    Yv = np.matrix(Yv)

    paramters = int(thetav.ravel().shape[1]) #[0,0,0,0,0]
    grad = np.zeros(paramters)

    error = sigmoid(Xv * thetav.T) - Yv #matrices

    for i in range(paramters):
        term = np.multiply(error, Xv[:,i]) #b-multiply l error ratio fy row by row 34an agib l total
        grad[i] = (alpha * np.sum(term))/ len(Xv)


    return grad



#minmize with function cost using gradient
result= opt.fmin_tnc(func=cost, x0 = theta,fprime=gradient,args=(X,Y))

print('result= ', result) #ht-print(thetas metrix, n. of trials,rkm m4 3rfah)

#cost after getting the best thetas
costafteroptimization  = cost(result[0],X,Y)
print('cost after optimization = ', costafteroptimization)


#calculate effeciency
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x > 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min,X)
#print('predictions =', predictions)

correct =[ 1 if ((a==1 and b==1) or (a==0 and b==0) ) else 0
           for(a,b) in zip(predictions,Y)] #

accuracy = ((sum(map(int,correct)) / len(correct))*100)
print('accuracy = {0}%'.format(accuracy))

