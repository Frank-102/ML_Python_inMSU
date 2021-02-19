import numpy as np
from numpy import array, dot, diag, reshape
import math
import csv
import matplotlib.pyplot as plt
import numpy.linalg as LA

#Part one: PCA
#dimension of PC
d = 2

#Reading the data file
csvFile = open("HW6_dat.csv", "r")
reader =csv.reader(csvFile)
xt = np.zeros((1000,2))  #builting a matrix to set the size
y = np.zeros(1000)   #y is the vectors of labels



for item in reader:
    if reader.line_num == 1:
        continue
        #ignore the first line.
    xt[reader.line_num -2][0]= item[0]
    #extract the the first number in second line to the matrix
    xt[reader.line_num -2][1]= item[1]
    y[reader.line_num - 2]= item[2]
X = np.transpose(xt) #building up the matrix X



color = []
for i in range(0,1000):
    if y[i] == 1:
        color.append("b")
    else:
        color.append("r")
for i in range(0,1000):
    plt.scatter(X[0][i], X[1][i], color=color[i])

plt.title("Original space")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()





#Part two: PCA

miu = np.zeros(2) #mean of X
for i in range(0,1000):
    miu[0] += (X[0][i]/1000)
    miu[1] += (X[1][i] / 1000)

X_tilde = np.zeros_like(X)

X_tilde[0]= X[0] - miu[0]      #centered Matrix X
X_tilde[1]= X[1] - miu[1]
XT_tilde = np.transpose(X_tilde)   #getting the transpose of X

Sigma = dot(X_tilde,XT_tilde)   #covariance matrix
lamda, u = LA.eig(Sigma)

if lamda[0] > lamda[1]:
    u1 = u[0]
    u2 = u[1]
else:
    u1 = u[1]
    u2 = u[0]



UT = np.zeros((2,2)) #orthonomal matrix
UT[0] = u1
UT[1] = u2

#plotting the bar
for i in range(0,1000):
   plt.scatter(X[0][i], X[1][i], color=color[i])

plt.title("Principal Component Analysis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


#Part Three and Question_2
def Radial_Basis_kernel(sigma, x, y): #kernel 1
    return math.exp(-LA.norm(x-y)*LA.norm(x-y)/2.0/sigma/sigma)

def Polynomial_kernel(a, x, y): #kernel 2
    return pow(dot(np.transpose(x),y),a)

#building the kernel matrix
K1 = np.zeros((1000,1000))
K2 = np.zeros_like(K1)
for i in range(1000):
    for j in range(1000):
        K2[i][j] = Polynomial_kernel(2,XT_tilde[i],XT_tilde[j])
        K1[i][j] = Radial_Basis_kernel(1,xt[i],xt[j])


h = np.identity(1000)-np.full((1000,1000),1/1000.0)
#centering matrix

Kernel_1_tilde = dot(dot(h,K1),h)
#centered kernel matrix
Kernel_2_tilde = dot(dot(h,K2),h)

w1,v1 = LA.eig(K1)
w2,v2 = LA.eig(K2)


v1T = np.transpose(v1)
v2T = np.transpose(v2)


#Kernel 1
lamda_1 = max(w1)
w1 = w1.tolist()
i = w1.index(lamda_1)
w1_ = w1
w1_[i] = -100
w1_ = np.asarray(w1_)
lamda_2 = max(w1_)
j = w1.index(lamda_2)
Lamda1 = np.zeros((2,2))

V1T = np.zeros((2,1000))

Lamda1[0][0] = np.sqrt(lamda_1)
Lamda1[1][1] = np.sqrt(lamda_2)



V1T[0] = v1T[i]
V1T[1] = v1T[j]
K1PCY = dot(Lamda1,V1T)
for i in range(0,1000):
    plt.scatter(K1PCY[0][i], K1PCY[1][i], color=color[i])
plt.title("Radial_Basis-uncentered K")
plt.xlabel("Kernel_RB_PC1")
plt.ylabel("Kernel_RB_PC2")
plt.show()


#Kernel2
lamda_1 = max(w2)
w2 = w2.tolist()


i = w2.index(lamda_1)
w2_ = w2
w2_[i] = -100
w2_ = np.asarray(w2_)


lamda_2 = max(w2_)
j = w2.index(lamda_2)
Lamda_2 = np.zeros((2,2))
V2T = np.zeros((2,1000))



Lamda_2[0][0] = np.sqrt(lamda_1)
Lamda_2[1][1] = np.sqrt(lamda_2)
V2T[0] = v2T[i]
V2T[1] = v2T[j]
Kernel_P_PCY = dot(Lamda_2,V2T)

for i in range(0,1000):
    plt.scatter(Kernel_P_PCY[0][i], Kernel_P_PCY[1][i], color=color[i])
plt.title("Polynomial-uncentered K")
plt.xlabel("Kernel_P_PC1")
plt.ylabel("Kernel_P_PC2")
plt.show()
