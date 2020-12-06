# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:21:40 2020

@author: hp
"""
import library as lib
import math

# calling the given function
f = lambda x,y: math.log(y)*y/x


# Inputs
print('Enter initial conditions:')
x0 = int(input('x0 = '))
y0 = float(input('y0 = '))
h  = float(input('h = '))

print('Enter final point:')
xn = int(input('end point, xn = '))

# Euler method call
lib.euler(f,x0,y0,h,xn)

"""
I have taken xn=8 and 4 step size h= 0.5, 0.2, 0.1, 0.02. We have initial
point x0=2 and y(2)=e=2.71828. I get the data to plot x_i Vs y(x_i) for different 
h values from this code. I have plotted all 4 graphs for different h value.
"""