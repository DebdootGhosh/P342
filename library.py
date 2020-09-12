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

def partialpivot(A):
    n=len(A)
    for r in range(n-1):
        if abs(A[r][r])==0:
            for i in range(r+1,n):
                if abs(A[i][r])>abs(A[r][r]):
                    A[i],A[r]=A[r],A[i]
                        
    return A                    
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
                    B[r],B[i]=B[i],B[r]
                    A[r],A[i]=A[i],A[r]
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

# forward substitution of lower triangular matrix with B
def forwardsubstitution(A,B): 
    n=len(A)
    lower,upper=luDecomposition(A, n)
    y=[0]*n
    for i in range(n):
        tmp = B[i]
        for j in range(i):
            tmp -= lower[i][j] * y[j]
        y[i] = tmp / lower[i][i] 
    return y

# backward substitution of upper triangular matrix with B
def backwardsubstitution(A,B):
    n=len(A)
    lower,upper=luDecomposition(A, n)
    x=[0]*n
    for i in range(n-1, -1, -1):
        tmp = B[i]
        for j in range(i+1, n):
            tmp -= upper[i][j] * x[j]
        x[i] = tmp / upper[i][i]
    return x

# main LU decomposition code
def LUsolution(A,B):
    n=len(A)
    lower,upper=luDecomposition(A, n)
    y=forwardsubstitution(lower,B)
    x=backwardsubstitution(upper,y)
    return x        