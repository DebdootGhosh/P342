# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 22:06:04 2021

@author: debdoot
"""

import matplotlib.pyplot as plt
import random as rd
import math

#random walk function in 2d so we have 0<=\theta_i <=pi to cover the whole plane . 
#Since each step is of length 1, for step i, Δxi=cos\theta_i and Δy=sin\theta_i
# So \theta_i=2*\pi*r_i where 0<=r_i<=1      
def randwalk(N):
    x=0
    y=0
    X=[]
    Y=[]
    #print('\n-----------RANDOM WALK-----------')
    #print('------------------------------')    
    #print('x   \ty   ')
    #print('------------------------------')
    for i in range(N):
        theta=2*math.pi*rd.random() 
        #print('%.5f\t%.5f\t'% (x,y) )
        X.append(x)
        Y.append(y)
        x+=math.cos(theta);          
        y+=math.sin(theta); 
    return(X,Y)

#for different random walks of same No. of steps
def randwalk_repeat(N,M):
    X=[]
    Y=[]
    for j in range(M):
        x,y = randwalk(N)
        X.append(x) 
        Y.append(y)
    return X,Y

#Plot some random walks for each step size
def plot_randomwalk(X,Y):
    for i in range(5):
        plt.title("plots of 5 Random Walks")  
        plt.xlabel("X")  
        plt.ylabel("Y") 
        plt.plot(X[i],Y[i],'o-', markersize='2',linewidth='1')
    plt.show()
    

# calculating average radial distance from the origin for different No. of step 
# calculate rms distance from origin for different No. of steps
#calculate average of  \Delta x, \Delta y

def Measurement(x,y):
    D=0
    R=0
    X=0
    Y=0
    for i in range(100):
        D+=((x[i][len(x[i])-1])**2 + (y[i][len(y[i])-1])**2)**0.5
        R+=((x[i][len(x[i])-1])**2 + (y[i][len(y[i])-1])**2)
        X=X+(x[i][len(x[i])-1])
        Y=Y+(y[i][len(y[i])-1])
    return D/100,(R/100)**0.5,X/100,Y/100

#calling different functions and using input No of steps we will get the answers
X1,Y1=randwalk_repeat(250,100)
X2,Y2=randwalk_repeat(500,100)
X3,Y3=randwalk_repeat(750,100)
X4,Y4=randwalk_repeat(1000,100)
X5,Y5=randwalk_repeat(1250,100) 

#plotting for different step no.
plot_randomwalk(X1, Y1) 
plot_randomwalk(X2, Y2)
plot_randomwalk(X3, Y3)
plot_randomwalk(X4, Y4)
plot_randomwalk(X5, Y5)

#printing results
a,b,c,d=Measurement(X1,Y1)
print("FOR STEPS NO.=250 ")
print("average radial distance is",a)
print("rms distance is",b)
print("average displacement in x direction is",c)
print("average displacement in y direction is",d)

a,b,c,d=Measurement(X2,Y2)
print("FOR STEPS NO.=500 ")
print("average radial distance is",a)
print("rms distance is",b)
print("average displacement in x direction is",c)
print("average displacement in y direction is",d)

a,b,c,d=Measurement(X3,Y3)
print("FOR STEPS NO.=750 ")
print("average radial distance is",a)
print("rms distance is",b)
print("average displacement in x direction is",c)
print("average displacement in y direction is",d)

a,b,c,d=Measurement(X4,Y4)
print("FOR STEPS NO.=1000 ")
print("average radial distance is",a)
print("rms distance is",b)
print("average displacement in x direction is",c)
print("average displacement in y direction is",d)

a,b,c,d=Measurement(X5,Y5)
print("FOR STEPS NO.=1250 ")
print("average radial distance is",a)
print("rms distance is",b)
print("average displacement in x direction is",c)
print("average displacement in y direction is",d)

#Plot Rrms Vs (N)^0.5
A=[15.645604735391055,21.784216599491128,26.735107062410364,32.29904420715503,36.763818075156]
B=[(250)**0.5,(500)**0.5,(750)**0.5,(1000)**0.5,(1250)**0.5]
plt.title("Rrms Vs (N)^0.5")  
plt.xlabel("Rrms")  
plt.ylabel("(N)^0.5") 
plt.plot(A,B,'o', markersize='4',label='observed data point')
plt.plot(A,A,'-',linewidth='2', label='analytical plot')
plt.legend()
plt.show()    


"""
FOR STEPS NO.=250 
average radial distance is 14.293099013097155
rms distance is 15.645604735391055
average displacement in x direction is -0.1371935120581815
average displacement in y direction is -0.8602060317825189

FOR STEPS NO.=500 
average radial distance is 20.329529606792665
rms distance is 21.784216599491128
average displacement in x direction is 0.4630022364200548
average displacement in y direction is 0.5088343348994291

FOR STEPS NO.=750 
average radial distance is 24.046154891277528
rms distance is 26.735107062410364
average displacement in x direction is 0.0013973397275966449
average displacement in y direction is 0.8256503094837138

FOR STEPS NO.=1000 
average radial distance is 29.128956970998765
rms distance is 32.29904420715503
average displacement in x direction is -0.5771554300282777
average displacement in y direction is 1.4769654072501197

FOR STEPS NO.=1250 
average radial distance is 31.78934251706527
rms distance is 36.763818075156
average displacement in x direction is 0.7657016983636877
average displacement in y direction is 0.99573260702312
"""