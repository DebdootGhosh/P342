# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 22:54:14 2020

@author: debdoot
"""
import library as lib
def luDecomposition(mat, n): 
  
    lower = [[0 for x in range(n)]  
                for y in range(n)]
    upper = [[0 for x in range(n)]  
                for y in range(n)]
                  
    # Decomposing matrix into Upper  
    # and Lower triangular matrix 
    for i in range(n): 
  
        # Upper Triangular 
        for j in range(i, n):  
  
            # Summation of L(i, k) * U(k, j) 
            sum = 0; 
            for k in range(i): 
                sum += (lower[i][k] * upper[k][j])
                
            # Evaluating U(i, j) 
            upper[i][j] = mat[i][j] - sum
  
        # Lower Triangular 
        for j in range(i, n): 
            if (i == j): 
                lower[i][i] = 1 # Diagonal as 1 
            else: 
  
                # Summation of L(j, k) * U(k, i) 
                sum = 0
                for k in range(i): 
                    sum += (lower[j][k] * upper[k][i])
  
                # Evaluating L(k, i) 
                lower[j][i] = ((mat[j][i] - sum) /
                                       upper[i][i])
        
   
    return lower,upper   

def forwordsubstitution(lower,B): 
    n=len(lower)
    y=[0]*n
    for i in range(n):
        tmp = B[i]
        for j in range(i):
            tmp -= lower[i][j] * y[j]
        y[i] = tmp / lower[i][i] 
    return y

def backwordsubstitution(upper,B):
    n=len(upper)
    x=[0]*n
    for i in range(n-1, -1, -1):
        tmp = B[i]
        for j in range(i+1, n):
            tmp -= upper[i][j] * x[j]
        x[i] = tmp / upper[i][i]
    return x

def LUsolution(lower,upper,B):
    y=forwordsubstitution(lower,B)
    x=backwordsubstitution(upper,y)
    return x
# Driver code 
A=open('iv.txt','r')
mat=lib.readwritematrix(A) 
  
l,u=luDecomposition(mat, 4)
print("lower triangular matrix:")
print(l)
print("upper triangular matrix:")
print(u)
B=[6,-3,-2,0]
lower,upper=luDecomposition(mat, 4)
m=LUsolution(lower,upper,B)
print("solution is:")
print(m)
y=forwordsubstitution(lower,B)
print("y is:")
print(y)
