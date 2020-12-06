# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math

def partialpivot(A,B):
    n=len(B)
    for r in range(n-1):
        if abs(A[r][r])==0:
            for i in range(r+1,n):
                if abs(A[i][r])>abs(A[r][r]):
                    for j in range(r,n):
                        A[r][j],A[i][j]=A[i][j],A[r][j]
                        B[r],B[i]=B[i],B[r]
    return B,A



def readwritematrix(fileA):
    A=[]
    for line in fileA.readlines():
        A.append([int (x) for x in line.split()])
    print(A)
    return A

             
def gaussjordan(A,B):
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
        print('Iteration-%d, x1 = %0.6f, absolute error = %f, and f(x1) = %0.6f' % (step, x1, abserror, f(x1)))
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
    print('\n--------SOLUTION---------------')
    print('--------------------------------')    
    print('x0  \ty0  \tyn ')
    print('--------------------------------')
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
        print('%.5f\t%.5f\t%.5f'% (x0,y0,yn) )
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


# lagrange interpolation function
def lagrange_interpolation(upper,lower,f1, f2, x0, y0, xn, yn, h):
    # yn for lower bracket
    yl = rk4(f2, x0, y0, lower,xn, h)    
    # yn for upper bracket
    yh = rk4(f2, x0, y0, upper, xn, h)    
    # for next y'(x0)
    g = lower + ((upper - lower) / (yh - yl)) * (yn - yl)
    # yn for the new y'(x0)
    y = rk4(f2, x0, y0, g, xn, h)
    print("Value of y(x=xn) found, integration successful")
    
        
