# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 07:58:19 2020

@author: debdoot
"""
import library as lib
import math

# coefficient array
a=[-18,27,-7,-3,1]
print("Solutions are:-")
#Calling Laguerre function from library
a=lib.Laguerre(a,4,1e-4,200)
a=lib.Laguerre(a,3,1e-4,200)
a=lib.Laguerre(a,2.5,1e-4,200)
a=lib.Laguerre(a,0,1e-4,200)

#Solutions are:-
#The root is: 3.0000260808250148
#Coefficients of the reduced polynomial are:
#[6.000052165731304, -6.9999217568447465, 2.608082501476261e-05, 1]
#The root is: 1.99993794713238
#Coefficients of the reduced polynomial are:
#[-3.000117804433029, 1.9999640279573947, 1]
#The root is: 1.0000415581627755
#Coefficients of the reduced polynomial are:
#[3.00000558612017, 1]
#The root is: -3.00000558612017
