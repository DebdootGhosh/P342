# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


def partialpivot1(A,B):
    n=len(B)
    for r in range(n-1):
        if abs(A[r][r])==0:
            for i in range(r+1,n):
                if abs(A[i][r])>abs(A[r][r]):
                    for j in range(r,n):
                        A[r][j],A[i][j]=A[i][j],A[r][j]
                        B[r],B[i]=B[i],B[r]
    return B,A
                    
def gaussjordan1(A,B): 
    n=len(B)
    for r in range(n):
        partialpivot(A,B)
        pivot=A[r][r]
        for c in range(r,n):
            A[r][c]=A[r][c]/pivot
        B[r]=B[r]/pivot
                            
        for i in range(n):
            if i==r or A[i][r]==0: continue 
            factor=A[i][r]
            for j in range(r,n):
                A[i][j]=A[i][j]-factor*A[r][j]
            B[i]=B[i]-factor*B[r]
                
    return B,A

def readwritematrix(fileA):
    A=[]
    for line in fileA.readlines():
        A.append([int (x) for x in line.split()])
    print(A)
    return A

def partialpivot2(A,B):
    n=len(B)
    for r in range(n-1):
        if abs(A[r][r])==0:
            for i in range(r+1,n):
                if abs(A[i][r])>abs(A[r][r]):
                    for j in range(r,n):
                        A[r][j],A[i][j]=A[i][j],A[r][j]
                        B[r][j],B[i][j]=B[i][j],B[r][j]
    return B,A
                    
def gaussjordan2(A,B):
    n=len(B)
    partialpivot(A,B)
    for r in range(n):
        pivot=A[r][r]
        for c in range(r,n):
            A[r][c]=A[r][c]/pivot
            B[r][c]=B[r][c]/pivot
                                
        for i in range(n):
            if i==r or A[i][r]==0: continue
            factor=A[i][r]
            for j in range(n):
                B[i][j]=B[i][j]-factor*B[r][j]
                A[i][j]=A[i][j]-factor*A[r][j]
                
    return B

def matrixmulti(M,N):
    
    result=[[0] * len(N[0]) for _ in range(len(M))]
    
    for i in range(len(M)): 
        for j in range(len(N[0])): 
            for k in range(len(N)): 
                result[i][j] =result[i][j] + M[i][k] * N[k][j] 
    return result    

def tvec(X):
    return[[i] for i in X]

def dotproduct(X,Y):
    dotproduct=0
    for i,j in zip(X,Y):
         dotproduct += i*j
    print('Dot product is : ' , dotproduct)

def vectoradd(X,Y):
    result=[]
    for i in range(0,len(X)):
        result.append(X[i]+Y[i])
    print('addition of the above two vectors is:',result)

def factorial(num):
    m=num
    factorial=1

    if num<0 :
        print("factorial of negative number does not exist")
    elif num==0:
        print("factorial of 0 is 1")
    else :
        while(num>0):
            factorial=factorial*num
            num=num-1;
        print("the factorial of", m, "is",factorial)
    
def sumofnaturalnum(num):
    m = num
    sum = 0

    if num <= 0: 
        print("enter a  positive number") 
    else: 
        while num > 0:
            sum = sum + num
            num = num - 1;
        print("Final Sum of first", m, "natural numbers is: ", sum) 