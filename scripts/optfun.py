"""
Test Functions For Optimization
https://www.sfu.ca/~ssurjano/optimization.html
"""
import numpy as np


def plot_point_on_fnc(fnc,coord):
    """
    draw a dot at given coordinates
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plot
    import numpy as np
    #%matplotlib inline 
    fig = plot.figure()
    ax = fig.gca(projection='3d')
    s = 0.05 # Try s=1, 0.25, 0.1, or 0.05
    X = np.arange(-2, 2.+s, s) #Could use linspace instead if dividing
    Y = np.arange(-2, 3.+s, s) #evenly instead of stepping... 
    #Create the mesh grid(s) for all X/Y combos.
    X, Y = np.meshgrid(X, Y)
    #Z =  X**2 + Y**2
    
    xx = np.stack((X,Y))
    Z =  fnc(xx)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
     linewidth=0, antialiased=False) #Try coolwarm vs jet
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plot.title('function ' + fnc.__name__)
    z_point = fnc(coord)
    ax.scatter(coord[0], coord[1], z_point+0.1, s = 500, c = 'g')
    plot.show()
    
#plotting a function of two variables, 3d picture
def plot_fnc(fnc):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plot
    import numpy as np
    #%matplotlib inline 
    fig = plot.figure()
    ax = fig.gca(projection='3d')
    s = 0.05 # Try s=1, 0.25, 0.1, or 0.05
    X = np.arange(-2, 2.+s, s) #Could use linspace instead if dividing
    Y = np.arange(-2, 3.+s, s) #evenly instead of stepping... 
    #Create the mesh grid(s) for all X/Y combos.
    X, Y = np.meshgrid(X, Y)
    #Z =  X**2 + Y**2
    
    xx = np.stack((X,Y))
    Z =  fnc(xx)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
     linewidth=0, antialiased=False) #Try coolwarm vs jet
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plot.title('function ' + fnc.__name__)
    plot.show()
 
#plot a function of two variables, given bounds on each variable, 3d picture    
def plot_fnc_rng(fnc, bounds = [[-2.,2.],[-2.,3]], s = 0.5):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plot
    import numpy as np
    #%matplotlib inline 
    fig = plot.figure()
    ax = fig.gca(projection='3d')
    s = 0.05 # Try s=1, 0.25, 0.1, or 0.05
    #X = np.arange(-2, 2.+s, s) #Could use linspace instead if dividing
    #Y = np.arange(-2, 3.+s, s) #evenly instead of stepping... 
    #Create the mesh grid(s) for all X/Y combos.
    rng = np.array(bounds)
    X = np.arange(rng[(0,0)], rng[(0,1)]+s, s) #Could use linspace instead if dividing
    Y = np.arange(rng[(1,0)], rng[1,1]+s, s) #evenly instead of stepping... 
  
    X, Y = np.meshgrid(X, Y)
    #Z =  X**2 + Y**2
    
    xx = np.stack((X,Y))
    Z =  fnc(xx)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
     linewidth=0, antialiased=False) #Try coolwarm vs jet
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plot.title('function ' + fnc.__name__)
    plot.show()  
    
def spher(x):
    '''
    smooth function of n variables, minimum = 0.0, achieved at (,0,..,0)
    '''
    sm = 0
    for i in range(x.shape[0]):
        sm+=x[i]**2
    return sm  

def booth(x): 
    '''
    smooth function of two variables, minimum = 0.0, achieved at (1,3)
    arg bounds = [(-10, 10), (-10, 10)]
    '''
    return (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2

def easom(x):        
    '''
    function of two variables, minimum = -1, achieved at (pi,pi)
    arg bounds = [(-20, 20), (-20, 20)]
    '''
    tmp = np.exp(-(x[0]-np.pi)**2 - (x[1]-np.pi)**2)
    
    return -np.cos(x[0])*np.cos(x[1])*tmp

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
    plot_fnc(spher)
    plot_fnc(booth)
    plot_fnc(ackley)
    plot_fnc(himmelblau)
    plot_fnc_rng(himmelblau)
    bounds = [[-2.,2.],[-2.,3]]
    #bounds = np.array(tmp)
    plot_fnc_rng(ackley,bounds)
    plot_fnc_rng(ackley)

    coord = [1.75, 1.75]
    plot_point_on_fnc(ackley,coord)
   
    bounds = [[-4.,4.],[-4.,5]]
    plot_fnc_rng(himmelblau,bounds)

