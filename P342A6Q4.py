# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:44:44 2020

@author: hp
"""

import library as lib
import math

# defining the function
f = lambda x: 4/(1+x**2)

print (" {:<8} {:<20} {:<10} ".format('i','pi','N'))

# results are
for i in range (1,401):
     p = lib.monte_carlo(f, 0, 1, 10*i)
     d = {i:[p, 10*i]}
     for  k, v in d.items():
          pi, N= v
          print ("{:<8} {:<20} {:<10} ".format(k, pi, N))   
          
          
          
          
          
        