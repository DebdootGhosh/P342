# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:52:48 2021

@author: hp
"""

import library as lib
import math

#defining the function under integration
#f = lambda x: 1/(1-4.69751*10**(-5)*(math.sin(x))**2)**0.5
f = lambda x: 1/(1-(math.sin(math.pi/8))**2 * (math.sin(x))**2)**0.5

#taking the integration of \phi. I have calculated the value of a first.
j = lib.simpson(f, 0, math.pi/2, 10)
print("result of the integration is ", j)

#I have multiplied the result of integration with a*(L/g)^0.5 value to get T
m=4*(1/9.8)**0.5*j
print("the value of time period T is:",m,"Second")

"""
result of the integration is  1.633586307458146
the value of time period T is: 2.087320017479592 Second
"""