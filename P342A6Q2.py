# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:58:07 2020

@author: hp
"""

import library as lib
import math

# defining the function
f = lambda x: x/(1+x)

# results are
p = lib.midpoint(f, 1, 3, 5)
print(" result of the integration is ", p)
 
a = lib.midpoint(f, 1, 3, 10)
print(" result of the integration is ", a)

b = lib.midpoint(f, 1, 3, 25)
print(" result of the integration is ", b)

c = lib.trapezoidal(f, 1, 3, 5)
print(" result of the integration is ", c)

d = lib.trapezoidal(f, 1, 3, 10)
print(" result of the integration is ", d)

e = lib.trapezoidal(f, 1, 3, 25)
print(" result of the integration is ", e)

i = lib.simpson(f, 1, 3, 5)
print(" result of the integration is ", i)

j = lib.simpson(f, 1, 3, 10)
print(" result of the integration is ", j)

k = lib.simpson(f, 1, 3, 25)
print(" result of the integration is ", k)

 #result of the integration is  1.308092114284065
 #result of the integration is  1.30716463959004
 #result of the integration is  1.3069028019555275
 #result of the integration is  1.3043650793650796
 #result of the integration is  1.3062285968245722
 #result of the integration is  1.306752839424082
 #result of the integration is  1.2084656084656085
 #result of the integration is  1.3068497693110697
 #result of the integration is  1.2869193931017784