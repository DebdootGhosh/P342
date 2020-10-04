# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:34:52 2020

@author: debdoot 
"""
import library as lib
import math
# given function
func = lambda x: math.log(x) - math.sin(x)
a=1.5
b=2.5
# calling bisection function from library
x= lib.bisection(func, a, b, tol=1e-6, maxit=200)
print("Solution is: x={},f(x)={}".format(x,func(x)))
# calling falseposition function from library
p= lib.falsePosition(func,a,b,1e-6,200)
# calling newtonRaphson function from library
i= lib.newtonRaphson(func,1.5,1e-6,200)

#Solution is: x=2.219106674194336,f(x)=-5.005784979861261e-07

#*** FALSE POSITION METHOD IMPLEMENTATION ***
#Iteration-1, x2 = 2.150691, absolute error = 2.150691, and f(x2) = -0.070732
#Iteration-2, x2 = 2.214279, absolute error = 0.063588, and f(x2) = -0.005084
#Iteration-3, x2 = 2.218778, absolute error = 0.004499, and f(x2) = -0.000347
#Iteration-4, x2 = 2.219085, absolute error = 0.000307, and f(x2) = -0.000024
#Iteration-5, x2 = 2.219106, absolute error = 0.000021, and f(x2) = -0.000002
#Iteration-6, x2 = 2.219107, absolute error = 0.000001, and f(x2) = -0.000000

#Required root is: 2.21910705


#*** NEWTON RAPHSON METHOD IMPLEMENTATION ***
#Iteration-1, x1 = 2.493456, absolute error = 2.493456, and f(x1) = 0.309968
#Iteration-2, x1 = 2.234774, absolute error = 0.258682, and f(x1) = 0.016593
#Iteration-3, x1 = 2.219175, absolute error = 0.015599, and f(x1) = 0.000072
#Iteration-4, x1 = 2.219107, absolute error = 0.000068, and f(x1) = 0.000000

#Required root is: 2.21910715
