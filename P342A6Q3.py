# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:38:37 2020

@author: hp
"""

import library as lib
import math

# defining the function
f = lambda x: math.exp(-x**2)
"""
I got |f"(x)|max=2 and |f""(x)|max=12 and we have b-a =1 and error =0.001.
After calculating n from the given formula of error for three methods, I got
n=9.128 for midpoint method so ceil(n)=10, n=12.90 for trapezoidal method so 
ceil(n)=13 and n=2.85 so ceil(n)=3 but as it is odd so I have taken n=4 for simpson method
"""
# results are
p = lib.midpoint(f, 0, 1, 10)
print(" result of the integration is ", p)
 
p = lib.trapezoidal(f, 0, 1, 13)
print(" result of the integration is ", p)
 
p = lib.simpson(f, 0, 1, 4)
print(" result of the integration is ", p)
 
#result of the integration is  0.7471308777479975
#result of the integration is  0.7464612610366896
#result of the integration is  0.7468553797909873
