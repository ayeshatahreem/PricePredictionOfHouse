import math
import numpy as np
import matplotlib.pyplot as plt
from pylab import xlabel, ylabel

def calMean(w):
    a = 0
    b = 0
    for i in range(w):
        a = a + float(area[i])
        b = b + float(bedrooms[i])
    a = a/w
    b = b/w
    return a,b

def calCost(x,y,theta,m):
   value = np.dot(x,theta) - y
   value2 = np.transpose(value)
   value = np.dot(value2,value)
   J = (1.0 / (2 * m)) * value
   return float(J)

def calCostOld(x,y,theta,m):
   b = 0;
   pred = x.dot(theta).flatten()
   b = b + (pred - y) ** 2
   J = (1.0 / (2 * m)) * b
   return J

def gradientDescent(x,y,theta, m, n, alpha):
   oldCost = []
   oldCost.append(calCost(x, y, theta, m))
   for i in range(n):
      d = 0
      e = 0
      f = 0
      pred = np.transpose(theta)
      pred = pred[0, 0] * x[:, 0] + pred[0, 1] * x[:, 1] + pred[0, 2] * x[:, 2]
      sum = pred - y
      for i in range(w):
          d=sum[i,0]* x[i, 0]+d
          e = sum[i, 0] * x[i, 1] + e
          f = sum[i, 0] * x[i, 2] + f
      a = theta[0, 0] - ( alpha * ( 1.0 / m ) * d)
      b = theta[1,0] - ( alpha * ( 1.0 / m ) * e)
      c = theta[2,0] - (alpha * (1.0 / m) * f)
      theta = []
      lista = []
      lista.append(a)
      lista.append(b)
      lista.append(c)
      lista=np.matrix(lista)
      lista=np.transpose(lista)
      theta=lista
      oldCost.append(calCost(x, y, theta,m))
   return theta, oldCost

def normalEq(x,y,theta):
    a=np.transpose(x)
    value = np.dot(a,x)
    value = np.linalg.pinv(value)
    value2 = np.dot(np.transpose(x), y)
    value = np.dot(value,value2)
    return value

def NEq(x,y,theta):
    a=np.transpose(x)
    value = np.dot(a,x)
    value = np.linalg.pinv(value)
    value2 = np.dot(value, a)
    value = np.dot(value2,y)
    return value

characteristics = 2;
f = open('ex1data2.txt')
area = []
bedrooms = []
price = []
w = 0
for line in f:
    w = w + 1
    i=line.find(',')
    area.append(line[0:i])
    j=line.find(',',i + 1)
    bedrooms.append(line[i + 1:j])
    k = line.find('\n', j + 1)
    price.append(line[j + 1:k])
f.close()
matrixX2 = []
for i in range(w):
    lista = []
    lista.append(1)
    lista.append(float(area[i]))
    lista.append(float(bedrooms[i]))
    matrixX2.append(lista)
matrixX2 = np.matrix(matrixX2)
a = 0
b = 0
a,b = calMean(w)
for i in range(w):
    area[i] = float(area[i])-a
    bedrooms[i] = float(bedrooms[i])-b
stdarea = 0
stdbedrooms = 0
for i in range (w):
    stdarea = float(area[i])*float(area[i]) + stdarea
    stdbedrooms = float(bedrooms[i])*float(bedrooms[i]) + stdbedrooms
stdarea = math.sqrt(stdarea/w)
stdbedrooms = math.sqrt(stdbedrooms/w)
for i in range(w):
    area[i] = float(area[i])/stdarea
    bedrooms[i] = float(bedrooms[i])/stdbedrooms
matrixX = []
matrixY = []
theta = []
for i in range(w):
    lista = []
    listb = []
    lista.append(1)
    lista.append(area[i])
    lista.append(bedrooms[i])
    listb.append(float(price[i]))
    matrixX.append(lista)
    matrixY.append(listb)
for i in range(characteristics+1):
    listtemp = []
    listtemp.append(0)
    theta.append(listtemp)
matrixX = np.matrix(matrixX)
matrixY = np.matrix(matrixY)
theta = np.matrix(theta)
iterations = 1500
alpha = 0.01
theta, J = gradientDescent(matrixX,matrixY,theta,w,iterations,alpha)
print(theta)
iterations2 =[]
J2 = []
for i in range(iterations):
    iterations2.append(str(i))
    J2.append(str((J[i])))
plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Regression with Multiple Variables")
xlabel("Iterations")
ylabel("Cost")
print(iterations2)
print(J2)
plt.plot(iterations2,J2)
plt.legend()
plt.show()
tArea=1650
tBedRooms=3
tArea=tArea-a
tBedRooms=tBedRooms-b
tArea=tArea/stdarea
tBedRooms=tBedRooms/stdbedrooms
price=theta[0,0]*1+theta[1,0]*tArea+theta[2,0]*tBedRooms
print("Gradient descent: ")
print(price)
theta = []
for i in range (characteristics+1):
    listtemp = []
    listtemp.append(0)
    theta.append(listtemp)
theta = np.matrix(theta)
theta = NEq(matrixX2,matrixY,theta)
print(theta)
tArea=1650
tBedRooms=3
price=theta[0,0]*1+theta[1,0]*tArea+theta[2,0]*tBedRooms
print("Normal eq: ")
print(price)