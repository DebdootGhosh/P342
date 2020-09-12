# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 20:17:58 2020

@author: debdoot
"""
#Q2
import library as lib
# read and write matrix
A=open('iv1.txt','r')
print("original matrix is:")
M=lib.readwritematrix(A) 
Q1=open('id.txt','r')
print("identity matrix is:")
Q=lib.readwritematrix(Q1) 
#partial pivoting 
N=lib.partialpivot(M)
#LU Decomposition and get the result
lower,upper=lib.luDecomposition(N,4)
print("lower matrix:")
print(lower)
print("upper matrix:")
print(upper)

def determinant(N):
    lower,upper=lib.luDecomposition(N,4)
    det=1
    for i in range(len(upper)):
        det=det*upper[i][i]
     
    return det    

a=determinant(N)
print("determinat of the N matrix is:")
print(a) 
if a==0:
    print("inverse of the matrix does not exist, it is singular")
else:
     print("inverse exists.") 
 
# inverse by LU decomposition method in order of partial pivoting
I1=lib.LUsolution(M,Q[1])  
I2=lib.LUsolution(M,Q[2]) 
I3=lib.LUsolution(M,Q[3]) 
I4=lib.LUsolution(M,Q[0]) 
I=[I1]+[I2]+[I3]+[I4]
print(I)
# get the inverse
def matrixtranspose(A):
    n =len(A)
    B = [[0 for x in range(n)]  
            for y in range(n)]
    for i in range(n):
        for j in range(n):
            B[i][j]=A[j][i]
    return B
P=matrixtranspose(I) 
print("inverse matrix is") 
for i in range(4): 
    for j in range(4): 
        print(P[i][j], " ", end='') 
    print()  
# checking the multiplication of inverse with the original matrix    
A=open('iv1.txt','r')
M=lib.readwritematrix(A)    
b=lib.matrixmulti(M,P)
print("matrix multiplication is:")
print(b)

#runfile('C:/Users/hp/Desktop/P342A4Q2.py', wdir='C:/Users/hp/Desktop')
#Reloaded modules: library
#original matrix is:
#[[0, 2, 8, 6], [0, 0, 1, 2], [0, 1, 0, 1], [3, 7, 1, 0]]
#identity matrix is:
#[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
#lower matrix:
#[[1, 0, 0, 0], [0.0, 1, 0, 0], [0.0, 0.0, 1, 0], [0.0, 0.5, -4.0, 1]]
#upper matrix:
#[[3, 7, 1, 0], [0, 2.0, 8.0, 6.0], [0, 0, 1.0, 2.0], [0, 0, 0, 6.0]]
#determinat of the N matrix is:
#36.0
#inverse exists.
#[[-0.25000000000000006, 0.08333333333333337, 0.16666666666666666, -0.08333333333333333], [1.6666666666666672, -0.666666666666667, -0.33333333333333326, 0.6666666666666666], [-1.8333333333333333, 0.8333333333333333, -0.3333333333333333, 0.16666666666666666], [0.3333333333333333, 0.0, 0.0, 0.0]]
#inverse matrix is
#-0.25000000000000006  1.6666666666666672  -1.8333333333333333  0.3333333333333333  
#0.08333333333333337  -0.666666666666667  0.8333333333333333  0.0  
#0.16666666666666666  -0.33333333333333326  -0.3333333333333333  0.0  
#-0.08333333333333333  0.6666666666666666  0.16666666666666666  0.0  
#[[0, 2, 8, 6], [0, 0, 1, 2], [0, 1, 0, 1], [3, 7, 1, 0]]
#matrix multiplication is:
#[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [4.163336342344337e-17, -3.3306690738754696e-16, 0.9999999999999999, 0.0], [2.7755575615628914e-17, -2.220446049250313e-16, -2.7755575615628914e-16, 1.0]]
