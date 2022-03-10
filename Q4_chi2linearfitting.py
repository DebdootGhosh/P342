# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:56:21 2022

@author: hp
"""
import library as lib
import matplotlib.pyplot as plt
import math

'''
# GIVEN X AND Y
x=[0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3,3.3]
y=[2.20,1.96,1.72,1.53,1.36,1.22,1.10,1.00,0.86,0.75,0.65,0.6]
#I have taken ln(y[i]) or ln(\omega)
z=[0.78,0.67,0.54,0.42,0.31,0.198,0.0953,0,-0.151,-0.287,-0.4307,-0.51]
'''
# FUNCTION
def func(x,a,b):
    A=[]
    for i in range(len(x)):
         m=0
         m=a+b*x[i]
         A.append(m)
    return A


# AVERAGE OF ELEMENTS OF X
def avg(x):
    A=0
    for i in range(len(x)):
        A=x[i]+A
    return A/len(x)

#SUM OF ELEMENT OF X
def sumv(x):
    S=0
    for i in range(len(x)):
        S=x[i]+S
    return S

# FOR x^a y^b type value generation
def power(x,a,y,b):
    z=[]
    for i in range(len(x)):
        z.append(pow(x[i],a)*pow(y[i],b))
    return z

# weight factor
def w(sig):
    w=[]
    for i in range(len(sig)):
        w.append((1/sig[i])**2)
    return w

def S(sig):
    v = w(sig)
    return sumv(v)

def SX(x, sig):
     v = w(sig)
     sx = power(x,1,v,1)
     return sumv(sx)
 
def SY(y, sig):
    v = w(sig)
    sy = power(y, 1, v, 1)
    return sumv(sy)
    
# SXX
def SXX(x, sig):
    v = w(sig)
    sxx=power(x, 2, v, 1)
    return sumv(sxx)

# SXY
def SXY(x,y,sig):
    v = w(sig)
    sxy=power(power(x,1,y,1),1,v,1)
    return sum(sxy)

#varience
def D(x,sig):
    return S(sig)*SXX(x,sig)-(SX(x,sig)**2)

# SLOPE CALCULATION
def slope(x,y,sig):
    s1 = S(sig)*SXY(x,y,sig)-SX(x,sig)*SY(y,sig)
    return s1/D(x,sig)

# INTERCEPT CALCULATION
def intercept(x,y,sig):
    S1=SXX(x,sig)*SY(y,sig)-SX(x,sig)*SXY(x,y,sig)
    return S1/D(x,sig)

    

#ERROR IN SLOPE    
def errslope(x,y,sig):
    return SXX(x,sig)/D(x,sig)
 
# ERROR IN INTERCEPT   
def errintercept(x,y,sig):
    return S(sig)/D(x,sig)

def cov_ab(x,sig):
    return -SX(x,sig)/D(x,sig)

#Person's r
#QUALITYFIT
def qualityfit(x,y,sig):
    return (SXY(x,y,sig))**2/(SXX(x,sig)*SXX(y,sig))

def Chi(x,y,sig):
    a = intercept(x, y, sig)
    b = slope(x, y, sig)
    z = func(x, a, b)
    z1 = lib.vector_subtraction(y, z)
    
    ch = power(z1, 2, w(sig), 1)
    return sumv(ch)/(len(x)-2)

A = open('msfit.txt','r')
m = lib.readwritematrix(A)
n = lib.transpose(m)
x = n[0]
y = n[1]
sig = n[2]
y1 =[math.log(y[i]) for i in range(len(y))]

sig1=[1/sig[i] for i in range(len(sig))]
'''
print(x)
print(y)
print(sig)
print(y1)
print(sig1)
'''
#print(y1)
b = slope(x, y1, sig1)
a = intercept(x, y1, sig1)
print('Chi^2/d.o.f:',Chi(x, y1, sig1))
print('quality fit:',qualityfit(x, y1, sig1))
print('slope= ', b)
print('intercept', a)
k = errslope(x,y1,sig1)
print('error in slope = sigma_n:',k)
print('error in intercept',errintercept(x,y1,sig1))

z = func(x,a,b)
#print(z)

plt.title("straight line fitting of question(ii)")  
plt.xlabel("t")  
plt.ylabel("ln(N)")
plt.plot(x,y1,'o', markersize='5',label='given data point')
plt.plot(x,z,'-', markersize='2',linewidth='2',label='fitted curve')
plt.legend()
plt.show()
c = math.e**a
#print('r0=',c)
#print('sigma_r0=', c*errintercept(x,y1,sig1))
t = abs(1/b)
print('life time of the given source:',t)
m = abs(k*t/b)
print('error in life time:', m/4)


