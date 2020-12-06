# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:04:38 2020

@author: hp
"""
import library as lib

# calling the given equation in function form where f(x,y,v)=d2y/dx2 or dv/dx
# and v=dy/dx
f = lambda x,y,v: 1-x-v

# Inputs
print('Enter initial conditions:')
x0 = float(input('x0 = '))
y0 = float(input('y0 = '))
v0=float(input('v0 = '))
print('Enter calculation point: ')
xn = float(input('xn = '))

print('Enter  step size:')
h = float(input('step size, h = '))

# RK4 method call
lib.rk4(f,x0,y0,v0,xn,h)

"""
I have taken xn=5 and 4 step size h= 0.5, 0.4, 0.2, 0.1. We have initial
point x0=0 and y(0)=2, y'(0)=1. I get the data to plot x_i Vs y(x_i) for different 
h values from this code. I have plotted all 4 graphs for different h value.Then 
I have got the analytical solution as 1+e^(-x)-(x^2)/2+2x and plotted it 
over the range x ∈ [−5, 5] and y ∈ [−5, 5] in the same graph to compair the 
plots with it.
"""