# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 07:58:19 2020

@author: debdoot
"""
import library as lib
import math


a=[-18,27,-7,-3,1]
print("Solutions are:-")
a=lib.Laguerre(a,4,1e-4,200)
a=lib.Laguerre(a,3,1e-4,200)
a=lib.Laguerre(a,2.5,1e-4,200)
a=lib.Laguerre(a,0,1e-4,200)

