# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:07:55 2022

@author: debdoot
"""

import math
from math import sqrt

#readwrite matrix function
def readwritematrix(fileA):
    A=[]
    for line in fileA.readlines():
        A.append([float (x) for x in line.split()])
    print(A)
    return A

# matrix subtraction function
def matrix_subtraction(A, B):
    """
    Subtracts matrix B from matrix A and returns difference
        :param A: The first matrix
        :param B: The second matrix
 
        :return: Matrix difference
    """
    # Section 1: Ensure dimensions are valid for matrix subtraction
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if rowsA != rowsB or colsA != colsB:
        raise ArithmeticError('Matrices are NOT the same size.')
 
    # Section 2: Create a new matrix for the matrix difference
    C = zeros_matrix(rowsA, colsB)
 
    # Section 3: Perform element by element subtraction
    for i in range(rowsA):
        for j in range(colsB):
            C[i][j] = A[i][j] - B[i][j]
 
    return C

# matrix addition function
def matrix_addition(A, B):
    """
    Adds two matrices and returns the sum
        :param A: The first matrix
        :param B: The second matrix
 
        :return: Matrix sum
    """
    # Section 1: Ensure dimensions are valid for matrix addition
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if rowsA != rowsB or colsA != colsB:
        raise ArithmeticError('Matrices are NOT the same size.')
 
    # Section 2: Create a new matrix for the matrix sum
    C = zeros_matrix(rowsA, colsB)
 
    # Section 3: Perform element by element sum
    for i in range(rowsA):
        for j in range(colsB):
            C[i][j] = A[i][j] + B[i][j]
 
    return C

# zero matrix function
def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
 
        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)
 
    return M

# identity matrix function
def identity_matrix(n):
    """
    Creates and returns an identity matrix.
        :param n: the square size of the matrix
 
        :return: a square identity matrix
    """
    IdM = zeros_matrix(n, n)
    for i in range(n):
        IdM[i][i] = 1.0
 
    return IdM

# transpose of matrix   
def transpose(M):
    """
    Returns a transpose of a matrix.
        :param M: The matrix to be transposed
 
        :return: The transpose of the given matrix
    """
    # Section 1: if a 1D array, convert to a 2D array = matrix
    if not isinstance(M[0],list):
        M = [M]
 
    # Section 2: Get dimensions
    rows = len(M)
    cols = len(M[0])
 
    # Section 3: MT is zeros matrix with transposed dimensions
    MT = zeros_matrix(cols, rows)
 
    # Section 4: Copy values from M to it's transpose MT
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]
 
    return MT 

# outer product function
def outer_product(A, B):
    n = len(A)
    m = len(B)
    C = [[0 for i in range(n)] for j in range(m)]
    for i in range(n):
        for j in range(m):
            total = 0
            for ii in range(1):
               total += A[i][ii] * B[j]
            C[i][j] = total
 
    return C

# print matrix function                        
def print_matrix(M, decimals=4):
    """
    Print a matrix one row at a time
        :param M: The matrix to be printed
    """
    for row in M:
        print([round(x,decimals)+0 for x in row])        
    return ''

#partial pivot fuction
'''
def partialpivot(A,B):
    n=len(B)
    for r in range(n-1):
        if abs(A[r][r])==0:
            for i in range(r+1,n):
                if abs(A[i][r])>abs(A[r][r]):
                    for j in range(r,n):
                        A[r][j],A[i][j]=A[i][j],A[r][j]
                        B[r][j],B[i][j]=B[i][j],B[r][j]
                        #B[r],B[i]=B[i],B[r]
    return A,B
'''
def parpivot(A):
    n=len(A)
    for r in range(n-1):
        if abs(A[r][r])==0:
            for i in range(r+1,n):
                if abs(A[i][r])>abs(A[r][r]):
                    for j in range(r,n):
                        A[r][j],A[i][j]=A[i][j],A[r][j]
                        
    return A

# gauss jordan function
def gaussjordan(a,b):
    n = len(b)
    
    # main loop
    for k in range(n):
        #partial pivoting
        if abs(a[k][k])<1.0e-12:
            for i in range (k+1,n):
                if abs(a[i][k])>abs(a[k][k]):
                    for j in range(k,n):
                        a[k][j],a[i][j] = a[i][j], a [k][j]
                    b[k],b[i] = b[i], b[k]
                    
                    break
        #division of the pivot row
        pivot = a[k][k]
        for j in range(k,n):
            a[k][j]/= pivot
        b[k]/= pivot
        #elimination loop
        
        for i in range(n):
            if i==k or a[i][k]==0: continue
            factor = a[i][k]
            for j in range(k,n):
                a[i][j]-= factor*a[k][j]
            b[i] -= factor*b[k]
            
    return b     

# multiplication of a vector and a matrix
def multiply(G, v):
    result = []
    for i in range(len(G)):
        r = G[i]
        total = 0
        for j in range(len(v)):
            total += r[j] * v[j]
        result.append(total)
    return result

# matrix multiplication
def matrix_multiply(A, B):
    """
    Returns the product of the matrix A * B
        :param A: The first matrix - ORDER MATTERS!
        :param B: The second matrix
 
        :return: The product of the two matrices
    """
    # Section 1: Ensure A & B dimensions are correct for multiplication
    rowsA = len(A)
    colsA = len(A[0])
    rowsB = len(B)
    colsB = len(B[0])
    if colsA != rowsB:
        raise ArithmeticError(
            'Number of A columns must equal number of B rows.')
 
    # Section 2: Store matrix multiplication in a new matrix
    C = zeros_matrix(rowsA, colsB)
    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total
 
    return C

# vector transpose
def tvec(X):
    return[[i] for i in X]

# dot product of two vector
def dotproduct(X,Y):
    dotproduct=0
    for i,j in zip(X,Y):
         dotproduct += i*j
    return dotproduct

# vector addition
def vector_add(X,Y):
    result=[]
    for i in range(0,len(X)):
        result.append(X[i]+Y[i])
    return result   

# vector subtraction
def vector_subtraction(X,Y):
    result=[]
    for i in range(0,len(X)):
        result.append(X[i]-Y[i])
    return result 

    
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

# LU decomposition function        
def luDecomposition(mat, n): 
  
    lower = [[0 for x in range(n)]  
                for y in range(n)]
    upper = [[0 for x in range(n)]  
                for y in range(n)]            
    # Decomposing matrix into Upper  
    # and Lower triangular matrix 
    parpivot(mat)
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
 
    
# Gauss-seidel function
# Defining our function as seidel which takes 3 arguments
# as A matrix, Solution and B matrix        
def seidel(a , b):
    n = len(a)                   
    x=[0 for i in range(n)]
    for k in range(1000):
        #print("ite",k+1,x)
        xnew = x[:]
        for j in range(0, n):        
            d = b[j]                  
            for i in range(0, n):     
                if(j != i):
                    d = d-a[j][i] * x[i] 
         # updating the value of our solution 
            xnew[j] = d / a[j][j]
        # returning our updated solution  
        
        normx = dotproduct(x, x)**.5
        normxn = dotproduct(xnew, xnew)**.5
        if  abs(normx-normxn)< 10**-5: 
            print('Iteration No.', j)
            return x
        x = xnew  
        
    return x    
        # returning our updated solution


# conjugate gradient method
def conjugate_grad(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    """
    n = len(b)
    if not x:
        x = [1 for _ in range(len(b))]
    r =vector_subtraction(multiply(A, x),b)
    p = [-r[i] for i in range(n)]
    r_k_norm = (dotproduct(r, r))
    for i in range(2*n):
        Ap = multiply(A, p)
        alpha = r_k_norm / dotproduct(p, Ap)
        x = [x[i]+(alpha * p[i]) for i in range(n)]
        r = [r[i]+(alpha * Ap[i]) for i in range(n)]
        r_kplus1_norm = (dotproduct(r, r))
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-8:
            print ('Itr:', i)
            break
        p = [(beta * p[i]) - r[i] for i in range(n)]
    return x    


# inverse of matrix
def inverse(A):
    n = len(A)
    x=[0 for i in range(n)]
    inv = []
    for i in range(n):
        x[i] = 1
        y = conjugate_grad(A, x)
        inv.append(y)
        x[i]=0
    return transpose(inv)     

# jacobi iteration method
def jacobi_iteration(A,b,N=10000,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed  
    n = len(A)                                                                                                                                                          
    if x is None:
        x = [0 for i in range(len(A))]

    # Create a diagonal matrix of the diagonal elements of A                                                                                                                                                
    # and subtract them from A        
    
    D = [[A[i][j] if i==j else 0 for j in range(n)]for i in range(n)]
    R = matrix_subtraction(A, D)
    #print(R)
    # Iterate for N times                                                                                                                                                                          
    for j in range(N):
        xnew=x[:]
        v = multiply(R,xnew)
        for i in range(n):
            xnew[i] = (b[i] - v[i]) / D[i][i]
            
        normx = dotproduct(x, x)**.5
        normxn = dotproduct(xnew, xnew)**.5
        if  abs(normx-normxn)< 10**-5: 
            print('Iteration No.', j)
            return x
        x = xnew    
    return x

# power iteration method for largest eigen value and corresponding eigen vector
def power_iteration(input_matrix, vector, error_tol = 1e-12, 
                    max_iterations = 100):


    # Ensure matrix is square.
    assert len(input_matrix[0]) == len(input_matrix[1])
    # Ensure proper dimensionality.
    assert len(input_matrix[0]) == len(vector)
    n = len(input_matrix[0])
    
    # Set convergence to False. Will define convergence when we 
    # exceed max_iterations
    # or when we have small changes from one iteration to next.

    convergence = False
    lamda_previous = 0
    iterations = 0
    error = 1e-12

    while not convergence:
        # Multiple matrix by the vector.
        w = multiply(input_matrix, vector)
        # Normalize the resulting output vector.
        norm_w = dotproduct(w, w)**.5
        # Normalize the resulting output vector.
        for i in range(n):
              vector[i] = w[i] / norm_w
        # Find rayleigh quotient
        # (faster than usual b/c we know vector 
        #is normalized already)
        lamda = dotproduct(vector, w)

        # Check convergence.
        error = abs(lamda - lamda_previous) / lamda
        iterations += 1

        if error <= error_tol or iterations >= max_iterations:
            convergence = True

        lamda_previous = lamda

    return lamda, vector

# power iteration method for 2nd largest eigen value and corresponding eigen vector
def power_iteration2(input_matrix, vector, error_tol = 1e-12, 
                    max_iterations = 100):
    n = len(input_matrix)
    lamda, vector = power_iteration(input_matrix, vector, error_tol = 1e-12, 
                    max_iterations = 100)
    
    norm_v = dotproduct(vector, vector)**.5
    for i in range(n):
              vector[i] = vector[i] / norm_v
    
    Vt = transpose(vector)
    UUt = outer_product(Vt, vector) 
    
    lUUt = [[0 for i in range(n)]  for j in range(n)]
    for i in range(n):
        for j in range(n):
              lUUt[i][j] = lamda*UUt[i][j] 
    A = [[0 for i in range(n)]  for j in range(n)]       
    for i in range(n):
          for j in range(n):
              A[i][j] = input_matrix[i][j] - lUUt[i][j]
    lamda2 , vector2 =  power_iteration(A, vector)
    
    return lamda2 , vector2
             
# jacobi method for eigen value and eigen vector              
def jacobi(a,tol = 1.0e-9): # Jacobi method

    def maxElem(a): # Find largest off-diag. element a[k,l]
        n = len(a)
        aMax = 0.0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(a[i][j]) >= aMax:
                    aMax = abs(a[i][j])
                    k = i; l = j
        return aMax,k,l
    
    def threshold(a):
        sum = 0.0
        for i in range(n-1):
            for j in range (i+1,n):
                sum = sum + abs(a[i][j])
        return 0.5*sum/n/(n-1)

    def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
        n = len(a)
        aDiff = a[l][l] - a[k][k]
        if abs(a[k][l]) < abs(aDiff)*1.0e-36: t = a[k][l]/aDiff
        else:
            phi = aDiff/(2.0*a[k][l])
            t = 1.0/(abs(phi) + sqrt(phi**2 + 1.0))
            if phi < 0.0: t = -t
        c = 1.0/sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = a[k][l]
        a[k][l] = 0.0
        a[k][k] = a[k][k] - t*temp
        a[l][l] = a[l][l] + t*temp
        for i in range(k):      # Case of i < k
            temp = a[i][k]
            a[i][k] = temp - s*(a[i][l] + tau*temp)
            a[i][l] = a[i][l] + s*(temp - tau*a[i][l])
        for i in range(k+1,l):  # Case of k < i < l
            temp = a[k][i]
            a[k][i] = temp - s*(a[i][l] + tau*a[k][i])
            a[i][l] = a[i][l] + s*(temp - tau*a[i][l])
        for i in range(l+1,n):  # Case of i > l
            temp = a[k][i]
            a[k][i] = temp - s*(a[l][i] + tau*temp)
            a[l][i] = a[l][i] + s*(temp - tau*a[l][i])
        for i in range(n):      # Update transformation matrix
            temp = p[i][k]
            p[i][k] = temp - s*(p[i][l] + tau*p[i][k])
            p[i][l] = p[i][l] + s*(temp - tau*p[i][l])
       
    n = len(a)
    maxRot = 5*(n**2)       # Set limit on number of rotations
    p = identity_matrix(n)  # Initialize transformation matrix
    for i in range(maxRot): # Jacobi rotation loop 
        aMax,k,l = maxElem(a)
        if aMax < tol: return [a[i][i] for i in range(n)],p
        rotate(a,p,k,l)
    print('jacobi does not converge')
    
    


'''        
# bisection method
def bisection(f, a, b, tol, maxit):
    
    fa=f(a)
    if abs(fa) < tol:
        return a
    
    fb=f(b)
    if abs(fb) < tol:
        return b
    
    if fa*fb > 0:
        for i in range(12):
            
            if abs(fa)<abs(fb):
                a = a - 1.5 * (b - a)
                return a,b
            if abs(fa)>abs(fb):
                b = b + 1.5 * (b - a)
                return a,b
        if i>12:
            print("Start with a new pair of (a,b)")
            return None
    c=0
    abserror=0
    for i in range(1,maxit+1):
        c_prev=c
        c = (a+b)/2
        abserror=abs(c-c_prev)
        fc=f(c)
        print('Iteration-%d, c = %0.6f, absolute error = %0.6f, and f(c) = %0.6f' % (i, c,abserror, f(c)))

        if abs(b-a) < tol:
            return
        if abs(fc)< tol:
            return c
        if fa*fc>0:
            a, fa=c, fc
        if fb*fc>0:
            b, fb= c, fc   
            
    return c          
 
 
# false Position method           
def falsePosition(f,x0,x1,e,N):
    if f(x0) * f(x1) > 0.0:
       print('Given guess values do not bracket the root.')
       print('Try Again with different guess values.')
    else:
      i = 1
      abserror=0
      x2=0
      print('\n\n*** FALSE POSITION METHOD IMPLEMENTATION ***')
      condition = True
      while condition and i <=N:
            x2prev=x2
            x2 = x0 - (x1-x0) * f(x0)/( f(x1) - f(x0) )
            abserror= abs(x2-x2prev)
            print('Iteration-%d, x2 = %0.6f, absolute error = %0.6f, and f(x2) = %0.6f' % (i, x2,abserror, f(x2)))

            if f(x0) * f(x2) < 0:
               x1 = x2
            else:
               x0 = x2

            i = i + 1
            condition = abs(f(x2)) > e

    print('\nRequired root is: %0.8f' % x2)
    if i > N:
        print('\nNot Convergent.')
    
# derivative function        
def derivativef(f,a,method):
    h=0.0001
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
        
      
#Newton Raphson method
def newtonRaphson(f,x0,e,N):
    print('\n\n*** NEWTON RAPHSON METHOD IMPLEMENTATION ***')
    step = 1
    condition = True
    x1=0
    abserror=0
    while condition and step <= N:
        if derivativef(f,x0,'central') == 0.0:
            print('Divide by zero error!')
            return
        x1prev=x1
        x1 = x0 - f(x0)/derivativef(f,x0,'central')
        abserror=abs(x1-x1prev)
        print('Iteration-%d, x1 = %0.8f, absolute error = %f, and f(x1) = %0.f' % (step, x1, abserror, f(x1)))
        x0 = x1
        step = step + 1
        condition = abs(f(x1)) > e
        
    print('\nRequired root is: %0.8f' % x1)
    if step > N:
        print('\nNot Convergent.')

#Laguerre method
# polynomial function
def polynomial(x,a):
    n=len(a)
    sum=0.0
    for i in range(n-1,-1,-1):
        sum+=a[i]*(x**i)
    return sum

# 1st derivative of polynomial
def d1polynomial(x,a):
    h=0.001
    y=(polynomial(x+h,a)-polynomial(x-h,a))/(2*h)
    return y

# 2nd derivative of polynomial
def d2polynomial(x,a):
    h = 0.001
    y = (polynomial(x + h, a) + polynomial(x - h, a)-2*polynomial(x,a)) / (2 * h*h)
    return y

# deflation or synthetic division
def deflation(root,a):
    n=len(a)
    A=[0 for i in range(n-1)]
    A[n-2]=a[n-1]
    #synthetic division
    for i in range(n-3,-1,-1):
        A[i]=a[i+1]+(root*A[i+1])

    return A

# Laguerre method
def Laguerre(a,x0,e,kmax):
    n=len(a)
    if n>2:
        m=x0
        mi,mj=x0,0
        k=1
        if polynomial(x0,a)!=0:
            while abs(mj-mi)>e and k<kmax:
                g=d1polynomial(m,a)/polynomial(m,a)

                h=g**2-(d2polynomial(m,a)/polynomial(m,a))

                deno1=g+math.sqrt((n-1)*(n*h-g**2))

                deno2=g-math.sqrt((n-1)*(n*h-g**2))

                if abs(deno1)>abs(deno2):
                    m=n/deno1
                else:
                    m=n/deno2

                if k%2==0:
                    mi=mj-m
                    m=mi

                else:
                    mj=mi-m
                    m=mj


                k+=1
        if k%2==0:
            print("The root is:",mi)
            a=deflation(mi,a)
        else:
            print("The root is:",mj)
            a = deflation(mj, a)
        print("Coefficients of the reduced polynomial are:")    
        print(a)
        return a 
    else:
        if a[1]*a[0]>0 :
            print("The root is:",-a[0]/a[1])
    
        else:
            print("The root is:",a[0]/a[1])
            

        return 0

# midpoint numerical integration method
def midpoint(f, a, b, n):
    # calculating step size
    h = (b-a)/n
    result = 0
    for i in range(n):
        result += f((a + h/2) + i*h)
    # Finding final integration value    
    result *= h
    return result

# trapezoidal numerical integration method
def trapezoidal(f, a, b, n):
    # calculating step size
    h = (b-a)/n
    # Finding sum
    result = 0.5*f(a)+0.5*f(b)
    for i in range(1,n):
        result += f(a + i*h)
    # Finding final integration value    
    result *= h 
    return result

# simpson numerical integration method
def simpson(f, a, b, n):
    # calculating step size
    h = (b - a) / n
    
    # Finding sum 
    result = f(a) + f(b)
    
    for i in range(1,n):
        x = a + i*h
        
        if i%2 == 0:
            result = result + 2 * f(x)
        else:
            result = result + 4 * f(x)
    
    # Finding final integration value
    result = result * h/3
    
    return result

# monte carlo integration method
import random   
def monte_carlo(f, a, b, n):
    x=[0 for i in range(n)]
    
    for i in range(n):
        x[i] = random.uniform(a,b)
        result=0.0
        
    for i in range(n):
        result += f(x[i])
        
    result *= (b-a)/n    
    
    return result    


# Euler method
def euler(f,x0,y0,h,xn):
    
    # Calculating no. of steps
    n= int ((xn-x0)/h)
    
    # printing the xi and y(xi) data 
    print('\n-----------SOLUTION-----------')
    print('------------------------------')    
    print('x0   \ty0   ')
    print('------------------------------')
    for i in range(n+1):
        slope = f(x0, y0)
        yn = y0 + h * slope
        print('%.5f\t%.5f\t'% (x0,y0) )
        y0 = yn
        x0 = x0+h

        
# RK-4 method
def rk4(f,x0,y0,v0,xn,h):
    
    # Calculating no. of steps 
    n = int((xn-x0)/h)
    # to solve dy/dx = v 
    def v(x,v):
        return v
    #print('\n--------SOLUTION---------------')
    #print('--------------------------------')    
    #print('x0  \ty0  \tyn ')
    #print('--------------------------------')
    # solving dv/dx = f(x,y,v) and dy/dx=v(x,v) simultaneously
    for i in range(n+1):
        k1 = h * (f(x0, y0,v0))
        k1v = h* (v(x0, v0))
        k2 = h * (f((x0+h/2), (y0+k1/2),v0))
        k2v = h* (v((x0+h/2),v0))
        k3 = h * (f((x0+h/2), (y0+k2/2),v0))
        k3v = h* (v((x0+h/2),v0))
        k4 = h * (f((x0+h), (y0+k3),v0))
        k4v = h* (v((x0+h),v0))
        k= (k1+2*k2+2*k3+k4)/6
        kv = (k1v+2*k2v+2*k3v+k4v)/6
        yn = y0 + kv
        vn = v0 + k
        #print('%.5f\t%.5f\t%.5f'% (x0,y0,yn) )
        #print('-----------------------------')
        y0 = yn
        x0 = x0+h
        v0 = vn
    return yn

#shooting method for boundary value problem
def shooting_method(f1,f2,x0,y0,xn,yn,h,g):
     # yn for the first guess
    y = rk4(f2, x0, y0, g, xn, h)
    print(f"Value of y(x=xn) for the above guess {g}=",y)

    # initialising
    lower,upper=0.0,0.0
    # checking if y overshoots or undershoots
    # if y overshoots
    if y > yn and abs(y - yn) > 10E-4:
        upper = g
        # we get upper bracket of y
        # to find lower bound
        while y > yn:
            g = float(input(f"Guess a value of y\'({x0}) lower than the previous guess\n"))
            y =rk4(f2, x0, y0, g, xn, h)
            print(f"Value of y(x=xn) for the above guess {g}=", y)
         # if yn for the guess is equal to or very near to actual yn   
        if abs(y - yn) < 10E-4:
            y=rk4(f2, x0, y0, g, xn, h)     
            print(f"Value of y(x=xn) for the above guess {g}=", y)
            print("Value of y(x=xn) found, integration successful")
            return 0
        else:                                # if yn of guess is less than actual yn
            lower = g                        # then we have found the lower bracket
            lagrange_interpolation(upper,lower,f1,f2,x0,y0,xn,yn,h)

        # if y undershoots
    elif y < yn and abs(y - yn) > 10E-4:
        lower = g     # got the lower bracket
        # now to find upper bound
        while y < yn:
            g = float(input(f"Guess a value of y\'({x0}) greater than the previous guess\n"))
            y = rk4(f2, x0, y0, g, xn,h)
            print(f"Value of y(x=xn) for the above guess {g}=", y)

        if abs(y - yn) < 10E-4:
            rk4(f2, x0, y0, g, xn, h)      
            print("Value of y(x=xn) found, integration successful")
        else:
            upper = g
            lagrange_interpolation(upper, lower, f1, f2, x0, y0, xn, yn, h)
        # if yn for the guess is equal to or very near to actual yn
    elif abs(y - yn) < 10E-4:           
        y=rk4( f2, x0, y0, g, xn, h)  
        print(f"Value of y(x=xn) for the above guess {g}=", y)
        print("Value of y(x=xn) found, integration successful")



def lagrange_interpolation(upper,lower,f1, f2, x0, y0, xn, yn, h):
    # yn for lower bracket
    yl = rk4(f2, x0, y0, lower,xn, h)    
    # yn for upper bracket
    yh = rk4(f2, x0, y0, upper, xn, h)    
    # for next y'(x0)
    g = lower + ((upper - lower) / (yh - yl)) * (yn - yl)
    # yn for the new y'(x0)
    y= rk4(f2, x0, y0, g, xn, h)
    print("Value of y(x=xn) found, integration successful")
    '''