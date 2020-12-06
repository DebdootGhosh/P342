# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:49:23 2020

@author: hp
"""
import library as lib

    
f1=lambda z,y,x:z
f2=lambda z,y,x:z+1
e = 2.71828
x0,y0=0,1
xn,yn=1,2*(e-1)

h=float(input("Enter step size h: \n"))
print("For h=",h)
print("-----------")
n = int((xn - x0) / h)
g = float(input(f"Guess a value of y\'({x0}) \n"))
lib.shooting_method(f1,f2,x0,y0,xn,yn,h,g)    
    
    
 
""" 
Enter the value of step size h: 
0.5
For h= 0.5
-----------

Guess a value of y'(0) 
11
Value of y(x=xn) for the above guess 11.0= 18.5625

Guess a value of y'(0) lower than the previous guess
1
Value of y(x=xn) for the above guess 1.0= 3.5625

Guess a value of y'(0) lower than the previous guess
0.9
Value of y(x=xn) for the above guess 0.9= 3.4124999999999996 
 
Enter step size h: 
0.4
For h= 0.4
-----------

Guess a value of y'(0) 
1
Value of y(x=xn) for the above guess 1.0= 2.84

Guess a value of y'(0) greater than the previous guess
1.3
Value of y(x=xn) for the above guess 1.3= 3.2

Guess a value of y'(0) greater than the previous guess
1.4
Value of y(x=xn) for the above guess 1.4= 3.4200000000000003

Value of y(x=xn) found, integration successful   


Enter step size h: 
0.2
For h= 0.2
-----------

Guess a value of y'(0) 
2
Value of y(x=xn) for the above guess 2.0= 4.220000000000001

Guess a value of y'(0) lower than the previous guess
1.2
Value of y(x=xn) for the above guess 1.2= 3.3600000000000002
Value of y(x=xn) found, integration successful


Enter step size h: 
0.1
For h= 0.1
-----------

Guess a value of y'(0) 
1.9
Value of y(x=xn) for the above guess 1.9= 3.832499999999999

Guess a value of y'(0) lower than the previous guess
1.3
Value of y(x=xn) for the above guess 1.3= 3.3725
Value of y(x=xn) found, integration successful
"""