# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:41:18 2021

@author: hp
"""

import library as lib

 # two function   
f1=lambda z,y,x:z
f2=lambda z,y,x:-9.8
#initial and final value
x0,y0=0,2
xn,yn=5,45

#input
h=float(input("Enter step size h: \n"))
print("For h=",h)
print("-----------")
n = int((xn - x0) / h)
g = float(input(f"Guess a value of y\'({x0}) \n"))
#calling shooting method fuction
lib.shooting_method(f1,f2,x0,y0,xn,yn,h,g) 
 
"""
I get the integration is succsesful  for initial guess 32 to 33. For 0.1,0.2 
step size it is 33. For 0.5 it is 32, for 0.4 it is 31. So initial velocity
dy/dt is nearly 33.
"""
    
"""   
Enter step size h: 
0.5
For h= 0.5
-----------

Guess a value of y'(0) 
40
Value of y(x=xn) for the above guess 40.0= 87.25000000000003

Guess a value of y'(0) lower than the previous guess
35
Value of y(x=xn) for the above guess 35.0= 59.75000000000002

Guess a value of y'(0) lower than the previous guess
33
Value of y(x=xn) for the above guess 33.0= 48.750000000000036

Guess a value of y'(0) lower than the previous guess
32
Value of y(x=xn) for the above guess 32.0= 43.25000000000003
Value of y(x=xn) found, integration successful

Enter step size h: 
0.2
For h= 0.2
-----------

Guess a value of y'(0) 
25
Value of y(x=xn) for the above guess 25.0= 4.599999999999967

Guess a value of y'(0) greater than the previous guess
30
Value of y(x=xn) for the above guess 30.0= 30.59999999999996

Guess a value of y'(0) greater than the previous guess
32
Value of y(x=xn) for the above guess 32.0= 40.99999999999996
Guess a value of y'(0) greater than the previous guess
33
Value of y(x=xn) for the above guess 33.0= 46.19999999999996
Value of y(x=xn) found, integration successful

Enter step size h: 
0.1
For h= 0.1
-----------

Guess a value of y'(0) 
30
Value of y(x=xn) for the above guess 30.0= 30.049999999999955

Guess a value of y'(0) greater than the previous guess
32
Value of y(x=xn) for the above guess 32.0= 40.24999999999994

Guess a value of y'(0) greater than the previous guess
33
Value of y(x=xn) for the above guess 33.0= 45.34999999999998
Value of y(x=xn) found, integration successful

Enter step size h: 
0.4
For h= 0.4
-----------

Guess a value of y'(0) 
35
Value of y(x=xn) for the above guess 35.0= 61.69599999999997

Guess a value of y'(0) lower than the previous guess
34
Value of y(x=xn) for the above guess 34.0= 56.49599999999998

Guess a value of y'(0) lower than the previous guess
33
Value of y(x=xn) for the above guess 33.0= 51.295999999999964

Guess a value of y'(0) lower than the previous guess
32
Value of y(x=xn) for the above guess 32.0= 46.09599999999997

Guess a value of y'(0) lower than the previous guess
31
Value of y(x=xn) for the above guess 31.0= 40.895999999999965
Value of y(x=xn) found, integration successful
"""