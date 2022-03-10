# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:54:45 2022

@author: hp
"""

import library as lib

b =[-1, 0, 2.75, 2.5, -3, 2]
A = open('msmatinvA.txt','r')
m = lib.readwritematrix(A)
print('The given A matrix:',lib.print_matrix(m))
print('print the b vector:', b)

print('Linear System solution by jacobi iteration method:',lib.jacobi_iteration(m, b))
print('Linear System solution by Gauss-Seidel iteration method:',lib.seidel(m, b))




