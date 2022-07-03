import minpy
import numpy as np
import copy
from optfun import spher,shor

class NM_Minimization(minpy.Minimization):
    
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
        count = 0
        while not self.stop() and count<=self.maxIter:
            self.update_c()
            self.estimate_simplex()
            self.update_m()
            self.adjust_step()
            count = count + 1
        
        return self.get_best()
    

X0 = np.array([0.8,1.9])

onSphere = NM_Minimization(spher,X0)

res = onSphere.NM_stochastic()

print('sphere optimal function value:', round(res[0],3), ' at X:', round(res[1][0],3)   ,round(res[1][1],3)) 
print('***********')

X0 = np.array([0.8,1.9,-0.5,1.2,2.1])

onShor = NM_Minimization(shor,X0)

res = onShor.NM_stochastic()
print('shor optimal function value:', round(res[0],3), ' at X:', round(res[1][0],3)   ,round(res[1][1],3),
     round(res[1][1],3), round(res[1][2],3), round(res[1][3],3), round(res[1][4],3)) 
print('***********') 
