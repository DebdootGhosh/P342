# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:17:48 2021

@author: hp
"""

import library as lib
import math
# GIVEN FUCTION
func = lambda x: (x-5)*((math.e)**x)+5

i= lib.newtonRaphson(func,4.2,1e-4,200)

# CALCULATION USING X VALUE FROM PREVIOUS SOLUTION AND OTHER GIVEN ELEMENT
h=6.626*10**(-34)
k=1.381*10**(-23)
c=3*10**8
x=4.96511425
b=(h*c)/(x*k)
print("Weine's constant is:",b,"meter-Kelvin")

"""
*** NEWTON RAPHSON METHOD IMPLEMENTATION ***
Iteration-1, x1 = 7.82511051, absolute error = 7.825111, and f(x1) = 7075.29862640
Iteration-2, x1 = 7.08601857, absolute error = 0.739092, and f(x1) = 2498.08411226
Iteration-3, x1 = 6.40870502, absolute error = 0.677314, and f(x1) = 860.23465393
Iteration-4, x1 = 5.82044669, absolute error = 0.588258, and f(x1) = 281.59112896
Iteration-5, x1 = 5.36161530, absolute error = 0.458831, and f(x1) = 82.04895187
Iteration-6, x1 = 5.07880275, absolute error = 0.282813, and f(x1) = 17.65427858
Iteration-7, x1 = 4.97689390, absolute error = 0.101909, and f(x1) = 1.64907849
Iteration-8, x1 = 4.96525381, absolute error = 0.011640, and f(x1) = 0.01930940
Iteration-9, x1 = 4.96511425, absolute error = 0.000140, and f(x1) = 0.00000274

Required root is: 4.96511425
Weine's constant is: 0.0028990103200792158 meter-Kelvin
"""