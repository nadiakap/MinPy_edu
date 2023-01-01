import numpy as np
import epydoc
import copy

class Trial(object):
    def __init__(self,u,fu,m):
        self.u = u
        self.fu = fu
        self.m = m
    

class Minimization(object):
    """
    This class contains building blocks for creating custom optimization algorithms
    Custom algorithms are defined in classes derived from this class
    
    the following parametrs are needed to initialize the algorithms
    
    f: callable - objective function for minimization, required parameter
    X0: np.array - initial guess for the minimum point (vector of dim elements), required parameter  
    dist: str - distribution type, default is 'uniform' 
    K: int -  number of random vectors, default is = 30
    m: np.array - center of mass of randomly generated vectors X, defaults to X0  
    c: float - mean level of objective function f, dedfaults to 0.0
    step_size: float - size of the step in the direction being explored on the given iteration of the algorithm, defaults to 0.5
    dim: int - dimensionality of the problem space (number of coordinates), defaults to X0.shape[0]
    tol: float - tolerance for accepting solution, defaults to 0.0001 
    u: np.array - array of K randomly generated vectors - a matrix of K rows and dim columns,initialized with [X0]
    fu: np.array - function values for randomly generated vectors u, initialized with np.zeros(K) 
    maxIter: int - maximum number of iterations of the algorithm, defaults to 600
    if K <= dim:
    K = dim + 1    
    lb: np.array -lower bound for the arguments, initialized with []
    ub: np.array -upper bound for the arguments, initialized with []
    """
    
    def __init__(self, f, X0, dist = 'uniform', K = 30, step_size = 0.5,
                 tol = 0.0001, maxIter = 600, line_search_type=None, lb = [], ub = []):
        self.f = f
        self.X0 = X0
        self.dist = dist
        self.K = K
        self.m = X0 
        self.c = 0.0
        self.step_size = 0.5 
        self.dim = X0.shape[0]
        self.tol = tol
        self.u = [X0] 
        self.fu = np.zeros(K)
        self.maxIter = maxIter
        if self.K <= self.dim:
            self.K = self.dim + 1    
        self.lb = lb
        self.ub = ub
        

    def initialize(self):
        np.random.seed(5)
        if self.dist == "exponential":
            self.u =  np.random.exponential(1.0,[self.K, self.dim])
        elif self.dist == "gaussian":
            self.u = np.random.normal(self.m, 1.0,[self.K,self.dim])
        else:
            self.u = np.random.rand(self.K,self.dim) 
            
            if len(self.lb)!=0 and len(self.ub)!=0:
                for i in range(self.dim):
                    self.u[:,i]=self.lb[i]+self.u[:,i]*(self.ub[i]-self.lb[i])
  
            else:
                for i in range(self.dim):
                    self.u[:,i]=self.m[i]-0.5+self.u[:,i]

    def compute_fu(self):
        for j in range(self.u.shape[0]):   
           self.fu[j] = self.f(self.u[j])
        
    def get_best(self):
        y=min(self.fu)
        idx = np.argmin(self.fu)
        x=self.u[idx]
        return (y,x)
        
    def sort(self):
        sorted_ind = np.argsort(self.fu)
        self.fu = np.take(self.fu, sorted_ind, 0)
        self.u = np.take(self.u,sorted_ind,0)

    
    def update_m(self):    
        self.m = np.mean(self.u[: -1],axis=0)
        
    def update_c(self):
        self.c = np.mean(self.fu)
    
    #sift out values lower than averge functino level
    def sift(self):
        self.fu = self.fu[self.fu<self.c]
        ind = np.argwhere(self.fu<self.c)
        self.u=self.u[ind]

    def update_simplex(self): 
         u_s = copy.copy(self.u ) 
         for i in range(self.K):
             u_s[i] = self.m + self.step_size*(self.fu[i]-self.c)*(self.m - self.u[i])/(np.linalg.norm(self.m-self.u[i])**self.dim)
         self.m = np.mean(u_s[: -1],axis=0)
 
    def reflect(self,z,rho = 1.): 
        return self.m + rho*(self.m-z)
    
    def contract(self,z,rho = 0.3): 
        return self.m + rho*(z - self.m)
    
    def shrink(self,z,rho,psi): 
        return (1-psi)*self.m + psi*z
    
    def expand(self,z,rho = 2.): 
        return self.m + rho*(z-self.m)
    
    def modify(self,z,s = 1.): 
        return self.m  + s*(self.m-z)   
    
    def compute_new_vertex(self,z,s): 
        fz = self.get_f(z)
        fm_tmp = self.get_f(self.m)
        step_size = s*(fz-fm_tmp)
        #vector of unit length
        direction = (self.m-z) /  np.linalg.norm(self.m-z)     
        return self.m  + step_size*direction
    
    def get_f(self,z):
        return self.f(z)
    
    def create_trial(self,trial_point):
        tr = Trial(self.u,self.fu,self.m)
        tr.u[-1] = trial_point
        tr.fu[-1] = self.get_f(trial_point)
        sorted_ind = np.argsort(tr.fu)
        tr.fu = np.take(tr.fu, sorted_ind, 0)
        tr.u = np.take(tr.u,sorted_ind,0)
        tr.m = np.mean(tr.u[: -1],axis=0)
        tr.fm = self.get_f(tr.m)     
        return tr
        
    def stop(self):
        return (sum((self.fu-self.c)**2  ))**0.5 < self.tol
    
    def adjust_step(self):
        pass
    
    def minimize_NMExt(self):
        self.initialize()
        self.compute_fu()
        self.sort() 
        self.update_m()
        newpoint = self.modify(self.u[-1],1.0)
        newpoint_f = self.get_f(newpoint)
        t = 0
        while not self.stop() and t<self.maxIter:

            if newpoint_f<self.fu[0]:
                reflection = newpoint
                reflection_f = newpoint_f
                newpoint = self.modify(self.u[-1],-2)
                newpoint_f = self.get_f(newpoint)
                if newpoint_f>reflection_f:
                    newpoint = reflection
                    newpoint_f = reflection_f

            elif newpoint_f>self.fu[-2]:
                newpoint = self.modify(newpoint,-0.5)
                newpoint_f = self.get_f(newpoint)

            if newpoint_f < self.fu[-2] : 
                self.u[-1] = newpoint
                self.fu[-1] = newpoint_f
                self.sort()
                self.update_m()
                newpoint = self.modify(self.u[-1],1.0)
                newpoint_f = self.get_f(newpoint)
            elif newpoint_f> self.fu[-1]:
                for i in range(1,len(self.fu)):
                    self.u[i]= self.u[0] + 0.1*(self.u[i]-self.u[0])
                self.compute_fu()
                self.sort()
                self.update_m()
                newpoint = self.modify(self.u[-1],1.0)
                newpoint_f = self.get_f(newpoint)
                
                
            t = t+1

      
        return (self.fu[0],self.u[0])
   

    def minimize_NM_global(self):
        fm = self.get_f(self.m)
        ftemp,futemp = self.minimize_NM()  
        while fm-ftemp>0.001:
            self.m = self.m/2
            ftemp,futemp = self.minimize_NM()
            if (np.max(np.ravel(np.abs(self.fu[1:] - self.fu[0]))) <= 0.001 and
                (np.max(np.abs(self.fu[0] - self.fu[1:])) <= 0.001)):
                    break 

        return (ftemp,futemp)
     
    def minimize_NM(self):
        self.initialize()
        self.compute_fu()
        self.sort() 
        self.update_m()
        newpoint = self.reflect(self.u[-1])
        newpoint_f = self.get_f(newpoint)
        t = 0
        while not self.stop() and t<self.maxIter:

            if newpoint_f<self.fu[0]:
                reflection = newpoint
                reflection_f = newpoint_f
                newpoint = self.expand(reflection)
                newpoint_f = self.get_f(newpoint)
                if newpoint_f>reflection_f:
                    newpoint = reflection
                    newpoint_f = reflection_f

            elif newpoint_f>self.fu[-2]:
                newpoint = self.contract(newpoint)
                newpoint_f = self.get_f(newpoint)

            if newpoint_f < self.fu[-2] : 
                self.u[-1] = newpoint
                self.fu[-1] = newpoint_f
                self.sort()
                self.update_m()
                newpoint = self.reflect(self.u[-1])
                newpoint_f = self.get_f(newpoint)
            elif newpoint_f> self.fu[-1]:
                for i in range(2,len(self.fu)):
                    self.u[i]= self.u[0] + 0.5*(self.u[i]-self.u[0])
                self.compute_fu()
                self.sort()
                self.update_m()
                newpoint = self.reflect(self.u[-1])
                newpoint_f = self.get_f(newpoint)
                
                
            t = t+1

      
        return (self.fu[0],self.u[0])
    
    def minimize_PSO_rr(self):
        self.initialize()
        self.compute_fu() 
        self.update_m()
        self.update_c()
        
        pbst=np.zeros((self.K, self.dim))
        gbst=np.zeros(self.dim)

         
        c0 = 0.9
        c1 = 0.25
        c2 = 0.25
        r1 = 1
        r2 = 1
        w = 0.9

        velo = 2*np.random.uniform(size=(self.K,self.dim))
            
        y_gbst = self.c
        gbst = self.m
        y_pbst = self.fu
        pbst = self.u
               
        #----moving forward
        for itim in range(self.maxIter):
            w = w * c0
            
            r1 = np.random.uniform()
            r2 = np.random.uniform()
            tmp1 = w * velo
            tmp2 = r1 * c1 * (pbst - self.u)
            tmp3 = r2 * c2 * (gbst - self.u)
            velo= tmp1 + tmp2 + tmp3
            tmp4 = self.u + velo
            self.u = tmp4
            self.compute_fu()
  
            for i in range(self.K):
                if self.fu[i] < y_pbst[i]:
                    y_pbst[i] = self.fu[i]
                    pbst[i] = self.u[i]
           
                if y_pbst[i] < y_gbst:
                    gbst = pbst[i]
                    y_gbst = y_pbst[i]
          
        return (y_gbst,gbst)     

if __name__ == "__main__": 
    
    import optfun as of
    
    X0 = np.array([0.8,1.9])

    myclass = Minimization(of.spher,X0)
    res = myclass.minimize_NM()
    print('sphere optimal function value:', round(res[0],3), ' at X:', round(res[1][0],3)   ,round(res[1][1],3)) 
    print('***********')
    myclass1 = Minimization(of.booth,X0)
    res1 = myclass1.minimize_NMExt()
    print('booth optimal function value:', round(res1[0],3), 'at X:', round(res1[1][0],3)   ,round(res1[1][1],3)) 
    print('***ackley optimization by NM_global algorithm ********')
    
    myclass4 = Minimization(of.ackley,X0)
    res4 = myclass4.minimize_NM_global()
    print('ackley optimal function value:', round(res4[0],3), ' at X:', round(res4[1][0],3)   ,round(res4[1][1],3),'with starting point ',X0) 
    X1 = np.array([1.9,-1.8])
    myclass5 = Minimization(of.ackley,X1)
    res5 = myclass5.minimize_NM_global()
    print('ackley optimal function value:', round(res5[0],3), ' at X:', round(res5[1][0],3)   ,round(res5[1][1],3),'with starting point ',X1) 
    X2 = np.array([-0.5,-0.8])
    myclass6 = Minimization(of.ackley,X2)
    res6 = myclass6.minimize_NM_global()
    print('ackley optimal function value:', round(res6[0],3), ' at X:', round(res6[1][0],3)   ,round(res6[1][1],3),'with starting point ',X2) 
    X4 = np.array([-1.9,1.8])
    myclass7 = Minimization(of.ackley,X4)
    res7 = myclass7.minimize_NM_global()
    print('ackley optimal function value:', round(res7[0],3), ' at X:', round(res7[1][0],3)   ,round(res7[1][1],3),'with starting point ',X4) 
    X3= np.array([-2.9,-2.5])
    myclass5 = Minimization(of.ackley,X3)
    res8 = myclass5.minimize_NM_global()
    print('ackley optimal function value:', round(res8[0],3), ' at X:', round(res8[1][0],3)   ,round(res8[1][1],3),'with starting point ',X3) 
    X5 = np.array([4.2,1.1])
    myclass10 = Minimization(of.ackley,X5)
    res10 = myclass10.minimize_NM_global()
    print('ackley optimal function value:', round(res10[0],3), ' at X:', round(res10[1][0],3)   ,round(res10[1][1],3),'with starting point ',X5) 
    X7 = np.array([4.7,4.8])
    myclass9 = Minimization(of.ackley,X7)
    res9 = myclass9.minimize_NM_global()
    print('ackley optimal function value:', round(res9[0],3), ' at X:', round(res9[1][0],3)   ,round(res9[1][1],3),'with starting point ',X7) 

    print('***********************')
    bounds = [(-5, 5), (-5, 5)]
    import scipy.optimize as spo
    resultAckley = spo.differential_evolution(of.ackley, bounds)
    print('ackley optimization by differential evolution algorithm:', resultAckley)
    print('***********************')

    print('***********************')
    bounds = [(-5, 5), (-5, 5)]
    import scipy.optimize as spo
    resultHimmelblau = spo.differential_evolution(of.himmelblau, bounds)
    print('himmelblau optimization by differential evolution algorithm:', resultHimmelblau)
    print('***********************')
    myclass = Minimization(of.booth,X0)
    res111 = myclass.minimize_NM()
    
    myclass = Minimization(spo.rosen,X0)
    res = myclass.minimize_NM()
    
    X13 = np.array([0.8,1.9])
    ac_lb=np.array([-5, -5])
    ac_ub=np.array([5, 5])
    myclass = Minimization(of.ackley,X13, K=100, lb=ac_lb,ub=ac_ub, maxIter=700)
    res = myclass.minimize_PSO_rr() 
    print('sphere optimization by PSO algorithm:', res)
    print('***********************')
 
    print('***********************')
    bounds = [(0, 10), (0, 10),(0, 10),(0, 10),(0, 10),(0, 10), (0, 10),(0, 10),(0, 10),(0, 10)]
    import scipy.optimize as spo
    resultOpt = spo.differential_evolution(of.mc_amer_opt, bounds)
    print('american option price  optimization by differential evolution algorithm:', resultOpt)