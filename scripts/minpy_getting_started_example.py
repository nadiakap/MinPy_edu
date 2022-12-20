import minpy
import numpy as np
import copy
from optfun import spher,himmelblau


    
class NM_Minimization(minpy.Minimization):
    
    @staticmethod
    def f_mean(f,u0,fu0,m0,c0,K0,dim0,step0):
        s = 0
        for i in range(K0):
             elem =  (fu0[i]-c0)*(m0 - u0[i])/(np.linalg.norm(m0-u0[i])**dim0)
             s = s + elem
        s=s*step0/K0
       
        m0 = m0 + s
        return f(m0)

    @staticmethod
    def get_step_size(f,f_m_ub,u0,fu0,m0,c0,K0,dim0):
            from scipy.optimize import golden
            def fma(step):
                return f_mean(f,u0,fu0,m0,c0,K0,dim0,step)-f_m_ub
            
            bracket= (0.02,0.5)
            return golden(fma,brack=bracket)     
                  
    def adjust_step(self, f_m_ub):
        self.step_size = get_step_size(self.f, f_m_ub, self.u,self.fu,self.m,self.c,self.K,self.dim)
               
    def estimate_mean(self):
        s = 0
        for i in range(self.K):
             elem =  (self.fu[i]-self.c)*(self.m - self.u[i])/(np.linalg.norm(self.m-self.u[i])**self.dim)
             s = s + elem
        s=s*self.step_size/self.K
        self.m = self.m + s
         
    def estimate_simplex(self): 
         u_s = copy.copy(self.u ) 
         for i in range(self.K):
             u_s[i] = self.m + self.step_size*(self.fu[i]-self.c)*(self.m - self.u[i])/(np.linalg.norm(self.m-self.u[i])**self.dim)

         self.m = np.mean(u_s[: -1],axis=0)
         
    def NM_stochastic(self):
        self.initialize()
        self.update_m()
        f_m_current = self.get_f(self.m)
        self.compute_fu()
        count = 0
        while not self.stop() and count<=self.maxIter:
            self.update_c()
            self.estimate_simplex()
            self.update_m()
        
            if self.get_f(self.m) >=f_m_current:
                self.adjust_step(f_m_current)
            count = count + 1
            f_m_current = self.get_f(self.m)
        return self.get_best()
    
X0 = np.array([0.8,1.9])

onSphere = NM_Minimization(spher,X0)

res = onSphere.NM_stochastic()

print('sphere optimal function value:', round(res[0],3), ' at X:', round(res[1][0],3)   ,round(res[1][1],3)) 
print('***********')

X0 = np.array([0.8,1.9,-0.5,1.2,2.1])

onHimmelblau = NM_Minimization(himmelblau,X0)

res = onHimmelblau.NM_stochastic()
print('himmelblau optimal function value:', round(res[0],3), ' at X:', round(res[1][0],3)   ,round(res[1][1],3),
     round(res[1][1],3), round(res[1][2],3), round(res[1][3],3), round(res[1][4],3)) 
print('***********') 
