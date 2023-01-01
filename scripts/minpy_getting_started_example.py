import minpy
import numpy as np
import copy
import numpy
from optfun import spher, himmelblau
from scipy.optimize import golden


class NM_Minimization(minpy.Minimization):
    """
    This class defines custom-made minimization algorithm that uses building blocks from minpy library
    Parametrs to all methods in this class are explained in Minimization class
    """
    
    @staticmethod
    def f_mean(f, u0, fu0, m0, c0, K0, dim0, step0):
        # comment on details
        s = 0.0
        for i in range(K0):
             elem =  (fu0[i]-c0)*(m0 - u0[i])/(np.linalg.norm(m0-u0[i])**dim0)
             s = s + elem
        s=s*step0/K0
       
        m0 = m0 + s
        return f(m0)

    @staticmethod
    def get_step_size(f, f_m_ub, u0, fu0, m0, c0, K0, dim0):

        def fma(step):
            return NM_Minimization.f_mean(f, u0, fu0, m0, c0, K0, dim0, step)-f_m_ub
        bracket = (0.02,0.5)
        return golden(fma,brack=bracket)     
                 
    def adjust_step(self, m_prev,f_m_prev):
        new_f_m = self.get_f(self.m)
        i = 0
        while (new_f_m >=f_m_prev and i <= 2):
            self.step_size = NM_Minimization.get_step_size(self.f, f_m_prev, self.u,self.fu,self.m,self.c,self.K,self.dim)
            i = i + 1
        if new_f_m >= f_m_prev:
            self.m = m_prev
        
    def estimate_mean(self):
        s = 0.0
        for i in range(self.K):
             elem =  (self.fu[i]-self.c)*(self.m - self.u[i])/(np.linalg.norm(self.m-self.u[i])**self.dim)
             s = s + elem
        s=s*self.step_size/self.K
        self.m = self.m + s
      
    def NM_stochastic(self):
        self.initialize()
        #compute center of mass self.m of the current simplex
        self.update_m()
        self.compute_fu()
        count = 0
        while not self.stop() and count<=self.maxIter:
            #store current value of center of mass
            m_prev = self.m
            #store function value at center of mass m_prev
            f_m_prev = self.get_f(m_prev)   
            self.update_c()
            self.update_simplex()
            self.update_m() 
            #adjusgt step size so that new center of mass is better than prev center of mass
            #if it cannot be achieved than set new center of mass at previos value
            self.adjust_step(m_prev,f_m_prev)
            count = count + 1
        return self.get_best()

    def NM_stochastic_from_Jupiter_notebook_example(self):
        self.initialize()
        self.update_m()
        while not self.stop():
            self.update_c()
            self.update_simplex()
            self.update_m()
            self.adjust_step()
        return self.get_best()

def main():
    """
    This is where the main code runs.
    """
    X0 = np.array([0.8,1.9])
    
    onSphere = NM_Minimization(spher,X0)
    
    res = onSphere.NM_stochastic()
    
    print('sphere optimal function value:', round(res[0],3), ' at X:', round(res[1][0],3)   ,round(res[1][1],3)) 
    print('***********')
    
    X0 = np.array([0.8,1.9,-0.5,1.2,2.1])
    
    onHimmelblau = NM_Minimization(himmelblau,X0, maxIter = 50)
    
    res = onHimmelblau.NM_stochastic()
    print('himmelblau optimal function value:', round(res[0],3), ' at X:', round(res[1][0],3)   ,round(res[1][1],3),
        round(res[1][1],3), round(res[1][2],3), round(res[1][3],3), round(res[1][4],3)) 
    print('***********') 

if __name__=='__main__':
    main()
