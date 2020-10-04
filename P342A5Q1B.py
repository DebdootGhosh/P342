# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 07:58:19 2020

@author: debdoot
"""
import library as lib
import math

func = lambda x: -x - math.cos(x)

a=-1
b=2
x= lib.bisection(func, a, b, 1e-6, 200)
print("Solution is: x={},f(x)={}".format(x,func(x)))

p= lib.falsePosition(func,a,b,1e-6,200)

i= lib.newtonRaphson(func,0,1e-6,200)

#Solution is: x=-0.7390847206115723,f(x)=-6.905382659017079e-07


#*** FALSE POSITION METHOD IMPLEMENTATION ***
#Iteration-1, x2 = -0.325149, absolute error = 0.325149, and f(x2) = -0.622455
#Iteration-2, x2 = -0.713324, absolute error = 0.388175, and f(x2) = -0.042868
#Iteration-3, x2 = -0.737776, absolute error = 0.024453, and f(x2) = -0.002190
#Iteration-4, x2 = -0.739020, absolute error = 0.001243, and f(x2) = -0.000110
#Iteration-5, x2 = -0.739082, absolute error = 0.000062, and f(x2) = -0.000006
#Iteration-6, x2 = -0.739085, absolute error = 0.000003, and f(x2) = -0.000000

#Required root is: -0.73908497


#*** NEWTON RAPHSON METHOD IMPLEMENTATION ***
#Iteration-1, x1 = -1.000000, absolute error = 1.000000, and f(x1) = 0.459698
#Iteration-2, x1 = -0.750364, absolute error = 0.249636, and f(x1) = 0.018923
#Iteration-3, x1 = -0.739113, absolute error = 0.011251, and f(x1) = 0.000046
#Iteration-4, x1 = -0.739085, absolute error = 0.000028, and f(x1) = 0.000000

#Required root is: -0.73908513