# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:23:28 2022

@author: hp
"""

import library as lib
import math

b = [1, 2, 0, 3, 5]
A = open('mstrimat.txt','r')
m = lib.readwritematrix(A)

p,q = lib.power_iteration(m, b)
print('The largest eigen value of the given matrix is:',p)
print('')
print('The eigen vector corresponding to the largest eigen value is:',q)
print('')
x,y = lib.power_iteration2(m, b)
print('The 2nd largest eigen value of the given matrix is:',x)
print('')
print('The eigen vector corresponding to the 2nd largest eigen value is:',y)

print('')
l1 = 2+2*math.sqrt(1)*math.cos(1*math.pi/6) 
print('The largest eigen value from the given expression:',l1)
print('')
l2 = 2+2*math.sqrt(1)*math.cos(2*math.pi/6) 
print('The 2nd largest eigen value from the given expression:',l2)

def eigenV_givExpre(k):
    A = [0 for i in range(5)]
    for i in range(5):
        v = 2*math.sin(i*k*math.pi/6)*(math.sqrt(1))**k
        A[i] = v
        
    return A  
#v = 2*math.sin(1*2*math.pi/6)*(math.sqrt(1))**2
j = eigenV_givExpre(2)  
print('eigen vector from given expression for largest eigen value',j)  
u = eigenV_givExpre(1)
print('eigen vector from given expression for largest eigen value',u)  
#print(v)
#The largest eigen value of the given matrix is: 3.7320508075620786

#The eigen vector corresponding to the largest eigen value is: [-0.2886764243233149, 0.500001289727799, -0.577350269187705, 0.49999871026887416, -0.2886738448643901]

#The 2nd largest eigen value of the given matrix is: 3.0000000000025295

#The eigen vector corresponding to the 2nd largest eigen value is: [0.4999994014520969, -0.4999989632459214, -1.197142310837108e-06, 0.5000010367386596, -0.5000005985590228]

#The largest eigen value from the given expression: 3.7320508075688776

#The 2nd largest eigen value from the given expression: 3.0
#eigen vector from given expression for largest eigen value [0.0, 1.7320508075688772, 1.7320508075688774, 2.4492935982947064e-16, -1.732050807568877]
#eigen vector from given expression for largest eigen value [0.0, 0.9999999999999999, 1.7320508075688772, 2.0, 1.7320508075688774]    
