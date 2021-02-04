# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:55:46 2021

@author: hp
"""
import matplotlib.pyplot as plt
import math
# GIVEN X AND Y
x=[0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3,3.3]
y=[2.20,1.96,1.72,1.53,1.36,1.22,1.10,1.00,0.86,0.75,0.65,0.6]
#I have taken ln(y[i]) or ln(\omega)
z=[0.78,0.67,0.54,0.42,0.31,0.198,0.0953,0,-0.151,-0.287,-0.4307,-0.51]

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

# fac1 function
def fac1(x):
    xx=power(x,2,y,0)
    xx1=avg(xx)
    x1x1=avg(x)*avg(x)
    return 1/(xx1-x1x1)

# delta function
def delta(x,y):
    D=[]
    for i in range(len(x)):
        D.append(x[i]-y[i])
    return D


# SXX
def SXX(x):
    sxx=0
    for i in range(len(x)):
        sxx=sxx+pow((x[i]-avg(x)),2)
    return sxx

# SXY
def SXY(x,y):
    sxy=0
    for i in range(len(x)):
        sxy=sxy+(x[i]-avg(x))*(y[i]-avg(y))
    return sxy

#STANDARD DEVIATION
def stDevx(x):
    return pow(SXX(x)/len(x),0.5)

# SLOPE CALCULATION
def slope(x,y):
    return SXY(x,y)/SXX(x)

# INTERCEPT CALCULATION
def intercept(x,y):
    return avg(y)-slope(x,y)*avg(x)

def S(x,y):
    return pow((SXX(y)-slope(x,y)*SXY(x,y))/(len(x)-2),0.5)

#ERROR IN SLOPE    
def errslope(x,y):
    return S(x,y)/pow(SXX(x),0.5)
 
# ERROR IN INTERCEPT   
def errintrecept(x,y):
    return S(x,y)*pow(((1/len(x))+((avg(x)**2)/SXX(x))),0.5)

#QUALITYFIT
def qualityfit(x,y):
    return (SXY(x,y)*SXY(x,y)/(SXX(x)*SXX(y)))**(1/2)

#OUTPUT
m1=(slope(x,y))
print('slope for question (i) fitting,\omega_c =',m1)
c1=(intercept(x,y))
print('y intersect for question (i) fitting,\omega_0',c1)

m2=(slope(x,z))
print('slope for question (ii) fitting,\omega_c=',m2)
c2=(intercept(x,z))
print('y intersect for question (ii) fitting,log(\omega_0) is',c2)
d=(pow(math.e,c2))
print("\omega_o for questin(ii)is",d)


print('Err slope for question (i) fitting,\delta omega_c is',errslope(x,y))
print('error in intersept for question (i) fitting,\delta \omega_0 is',errintrecept(x,y))


print('Err slope for question (ii) fitting,\delta \omega_c is'  ,errslope(x,z))
print('error in intersept for question (ii) fitting is',errintrecept(x,z))


print('quality of square fit for question (i) fitting is',qualityfit(x,y))
print('quality of square fit for question (ii) fitting is',qualityfit(x,z))

n1=func(x,c1,m1)
#print(n1)
plt.title("straight line fitting of question(i)")  
plt.xlabel("TIME")  
plt.ylabel("OMEGA") 
plt.plot(x,y,'o', markersize='5',label='given data point')
plt.plot(x,n1,'-', markersize='2',linewidth='2',label='fitted curve')
plt.legend()
plt.show()

n2=func(x,c2,m2)
#print(n2)
plt.title("straight line fitting of question(ii)")  
plt.xlabel("TIME")  
plt.ylabel("LOG(OMEGA)") 
plt.plot(x,z,'o', markersize='5',label='given data point')
plt.plot(x,n2,'-', markersize='2',linewidth='2',label='fitted curve')
plt.legend()
plt.show()

"""
slope for question (i) fitting,\omega_c = -0.47470862470862485
y intersect for question (i) fitting,\omega_0 2.0291025641025646

slope for question (ii) fitting,\omega_c= -0.39362470862470866
y intersect for question (ii) fitting,log(\omega_0) is 0.785697435897436
\omega_o for questin(ii)is 2.193936537371271

Err slope for question (i) fitting,\delta omega_c is 0.026157630469407393
error in intersept for question (i) fitting,\delta \omega_0 is 0.05095705145102684

Err slope for question (ii) fitting,\delta \omega_c is 0.005373193131058415
error in intersept for question (ii) fitting is 0.010467388441620212

quality of square fit for question (i) fitting is 0.9851557666128383
quality of square fit for question (ii) fitting is 0.9990696126872172
"""