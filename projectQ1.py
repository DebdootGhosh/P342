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
    for j in range(M):
        x,y = randwalk(N)
          
    
#Plot some random walks for each step size
def plot_randomwalk(N,M):
    for j in range(M):
        x,y = randwalk(N)
        plt.title("plots of 5 Random Walks")  
        plt.xlabel("X")  
        plt.ylabel("Y") 
        plt.plot(x,y,'o-', markersize='2',linewidth='1')
    plt.show()    
    
 # calculating radial distance from the origin for different No. of step   
def Radial_distance(N,M):
    D=0
    for i in range(M):
        x,y=randwalk(N)       
        D+=((x[N-1])**2 + (y[N-1])**2)**0.5
    return D/i 

# calculate rms distance from origin for different No. of steps
def RMS_distance(N,M):
    R=0
    for i in range(M):
        x,y=randwalk(N)       
        R+=((x[N-1])**2 + (y[N-1])**2)
    return (R/i)**0.5

#calculate average of 
def AverageXandY(N,M):
    X=0
    Y=0
    for i in range(M):
        x,y=randwalk(N)
        X=X+(x[N-1])
        Y=Y+(y[N-1])
    return X/i,Y/i

#calling different functions and using input No of steps we will get the answers 
N=int(input("Enter No. of steps N: \n"))   
print("For N=",N)
randwalk_repeat(N,100)
plot_randomwalk(N,5) 
y=Radial_distance(N, 100)
print("radial distance is",y)
z=RMS_distance(N, 100)
print("rms distance is",z)
a,b=AverageXandY(N, 100) 
print("average displacement in x direction is",a)
print("average displacement in y direction is",b)

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

#Enter No. of steps N: 
#250
#For N= 250
#radial distance is 14.95640542317947
#rms distance is 15.645604735391055
#average displacement in x direction is -0.34453918159327634
#average displacement in y direction is -0.11941714354508191


#Enter No. of steps N: 
#500
#For N= 500
#radial distance is 20.00829107318204
#rms distance is 21.784216599491128
#average displacement in x direction is 0.15984040514418113
#average displacement in y direction is -0.24852430060212818


#Enter No. of steps N: 
#750
#For N= 750
#radial distance is 26.084281002043596
#rms distance is 26.735107062410364
#average displacement in x direction is 0.36154119169132765
#average displacement in y direction is -0.23895645556894482


#Enter No. of steps N: 
#1000
#For N= 1000
#radial distance is 28.55942296942913
#rms distance is 32.29904420715503
#average displacement in x direction is 0.5836234543098227
#average displacement in y direction is -0.5585448841522804


#Enter No. of steps N: 
#1250
#For N= 1250
#radial distance is 30.893012515829536
#rms distance is 36.763818075156
#average displacement in x direction is 0.18888165153457442
#average displacement in y direction is 0.4211936998385299

