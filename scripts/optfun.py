import numpy as np
def spher(x):
    '''
    smooth function of n variables, minimum = 0.0, achieved at (,0,..,0)
    '''
    sm = 0
    for i in range(x.shape[0]):
        sm+=x[i]**2
    return sm**0.5  

def booth(x): 
    '''
    smooth function of two variables, minimum = 0.0, achieved at (1,3)
    arg bounds = [(-10, 10), (-10, 10)]
    '''
    return (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2

def easom(x):        
    '''
    function of two variables, minimum = -1, achieved at (pi,pi)
    arg bounds = [(-100, 100), (-100, 100)]
    '''
    return -np.cos(x[0])*np.cos(x[1])*np.exp(-((x[0]-np.pi)**0.5+(x[1]-np.pi)**0.5))

def ackley(x):
    '''
    non smooth function of two variables, minimum = 0.0, achieved at (,0,..,0)
    arg bounds = [(-5, 5), (-5, 5)]
    '''

    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

def himmelblau(x):
    '''
    multi-modal function of two variables. It has one local maximum f(x,y)=181.617, 
    achieved at x=-0.270845 and y=-0.923039,'
    and four identical local minima = 0.0, achieved at (3.0,2.0), (-2.805118,3,131312),
    (-3.779310,-3.283186),(3.584428,-1.848126)
 
    '''

    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

#euroean option proce by monte carlo, as a function of 
def mc_amer_opt(x):
    import random as rnd
    import scipy.stats as sst
    from math import exp,sqrt
    S0 = 100
    K = 98
    r = 0.03
    t = 1
    npd = 10
    npth = 1000
    vol = 0.3

    dt = t / npd
    
    rnd.seed(99)

    copt = 0
    #ceur = 0
    drft = (r - 0.5 * vol ** 2) * dt
    for ipth in range(npth):
        stk_ij = S0
        optj = 0
        for ipd in range(1,npd):
            u = np.random.uniform()           
            dftn = vol * sqrt(dt) * sst.norm.ppf(u)
            stk_ij = stk_ij * exp(drft + dftn)
            if stk_ij < x[ipd]:
                pay = max(K - stk_ij, 0)
                optj = exp(-r * ipd * dt) * pay
                break
           
        copt = copt + optj

    return copt / npth

    
def shor(x,a,b):
    '''
    non smooth function
    
    '''
    mx = 0
    m = a.shape[1]
    n = a.shape[0]
    print(m,n)
    phi = 0
    
    for j in range(n):
            phi+=(x[j]-a[j][0])**2
    mx  =  phi*b[0]
    
    for i in range(m):
        phi = 0
        for j in range(1,n):
            phi+=(x[j]-a[j][i])**2
        f  =  phi*b[i] 
        print(f)
        if f>mx:
            mx=f
           
    return mx

def shor10(x):
    '''
    non smooth function in 10 dim space
    
    '''
    a=np.array([[ 0,  2,  1,  1,  3,  0,  1,  1,  0,  1],   
       [0,  1,  2,  4,  2,  2,  1,  0,  0,  1],   
       [0,  1,  1,  1,  1,  1,  1,  1,  2,  2],   
       [0,  1,  1,  2,  0,  0,  1,  2,  1,  0],   
       [0,  3,  2,  2,  1,  1,  1,  1,  0,  0]] )
    b=np.array([ 1, 5, 10, 2, 4, 3, 1.7, 2.5, 6, 4.5])
    '''
    a1=np.array([[ 0,  2],   
       [0,  1],   
       [0,  1],   
       [0,  1],   
       [0,  3]]) 
    b1=np.array([ 1, 5])
    '''

    mx = 0
    m = a.shape[1]
    n = a.shape[0]
  
    phi = 0
    
    for j in range(n):
            phi+=(x[j]-a[j][0])**2
    mx  =  phi*b[0]
    
    for i in range(m):
        phi = 0
        for j in range(1,n):
            phi+=(x[j]-a[j][i])**2
        f  =  phi*b[i] 
        
        if f>mx:
            mx=f
           
    return mx

if __name__== "__main__":
    from algopy import UTPM
    arg = [0.2,1.1,3.2,1.2,-0.8,1.2,-1.2,0.9,-0.87,-0.5]
    y = shor10(arg)
    print('shor10_value = ',y)
    x = UTPM.init_jacobian(arg)
    y = shor10(x)
    algopy_jacobian = UTPM.extract_jacobian(y)    
        