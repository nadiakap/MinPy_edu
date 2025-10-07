import numpy as np
#import epydoc
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
    
    def minimize_potential_nonlocal(self,
                                kernel='inv_power',
                                kernel_param=2.0,
                                eps0=1.0,
                                eps_final=1e-3,
                                eps_anneal_rate=0.9,
                                beta0=1.0,
                                beta_final=20.0,
                                beta_anneal_rate=1.05,
                                step=0.1,
                                max_epochs=200,
                                r_cut=None,
                                neighbor_limit=None,
                                replace_worst_frac=0.2,
                                local_refine=False,
                                verbose=False):
            """
            Nonlocal potential-based minimizer inspired by Kaplinskii & Propoi.
            Builds a potential field from current samples and moves particles along -grad Phi.
        
            Parameters
            ----------
            kernel : {'inv_power','inv_multiquadric','gaussian'}
                Kernel type used to build potential.
            kernel_param : float
                Parameter for kernel (power exponent for 'inv_power', sigma for 'gaussian').
            eps0 : float
                Initial regularization / smoothing scale (large => more global).
            eps_final : float
                Final smoothing scale (small => local refinement).
            eps_anneal_rate : float
                Multiplicative factor to reduce eps each epoch (0.9 means eps *= 0.9).
            beta0 : float
                Initial inverse-temperature for weight exp(-beta f).
            beta_final : float
                Final beta (higher => weights concentrate on best points).
            beta_anneal_rate : float
                Multiplicative increase each epoch (e.g. 1.05).
            step : float
                Movement step multiplier applied to computed forces.
            max_epochs : int
                Maximum outer iterations.
            r_cut : float or None
                Optional truncation radius; if set, ignore interactions with r > r_cut.
            neighbor_limit : int or None
                If set, only the nearest `neighbor_limit` samples contribute to force (approx).
            replace_worst_frac : float in [0,1]
                Fraction of worst particles to re-sample around best ones each epoch.
            local_refine : bool
                If True, run scipy.optimize.minimize (Nelder-Mead) on best point at end (optional).
            verbose : bool
                Print diagnostics.
        
            Returns
            -------
            (best_f, best_x)
            """
            import math
            # optional imports (used only if available)
            try:
                from scipy.spatial import cKDTree as KDTree
            except Exception:
                KDTree = None
            try:
                from scipy.optimize import minimize as sp_minimize
            except Exception:
                sp_minimize = None
        
            # helpers: kernels and their radial derivatives dG/dr
            def kernel_G_and_dG(r, eps):
                # r: array
                if kernel == 'inv_power':
                    # G(r) = 1 / ( (r^p) + eps ), p = kernel_param
                    p = float(kernel_param)
                    rp = np.power(r + 0.0, p)  # r^p
                    denom = rp + eps
                    G = 1.0 / denom
                    # dG/dr = - p * r^{p-1} / (rp + eps)^2
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dG = - p * np.power(r, max(p-1, 0.0)) / (denom * denom)
                    # handle r==0: set derivative to 0 (no self-force)
                    dG = np.where(r == 0.0, 0.0, dG)
                    return G, dG
                elif kernel == 'inv_multiquadric':
                    # G = 1/sqrt(r^2 + eps)
                    denom = np.sqrt(r*r + eps)
                    G = 1.0 / denom
                    # dG/dr = - r / (r^2 + eps)^(3/2)
                    dG = - r / np.power(r*r + eps, 1.5)
                    dG = np.where(r == 0.0, 0.0, dG)
                    return G, dG
                elif kernel == 'gaussian':
                    sigma = float(kernel_param)
                    G = np.exp(-0.5 * (r / sigma)**2)
                    dG = (-r / (sigma**2)) * G
                    dG = np.where(r == 0.0, 0.0, dG)
                    return G, dG
                else:
                    raise ValueError("Unknown kernel: " + str(kernel))
        
            # ensure u is an array shape (K, dim)
            self.u = np.asarray(self.u, dtype=float)
            if self.u.ndim == 1:
                self.u = self.u[np.newaxis, :]
            K = self.u.shape[0]
            dim = self.dim
        
            # evaluate fu if not present
            if not hasattr(self, 'fu') or len(self.fu) != K:
                self.fu = np.zeros(K)
                for i in range(K):
                    self.fu[i] = self.get_f(self.u[i])
        
            # keep track of best
            best_idx = np.argmin(self.fu)
            best_x = self.u[best_idx].copy()
            best_f = self.fu[best_idx]
        
            eps = float(eps0)
            beta = float(beta0)
        
            # Precompute index array for excluding self interactions
            idxs = np.arange(K)
        
            # main loop
            for epoch in range(max_epochs):
                # compute weights: larger for small f
                # use numerically stable softmax-like scaling
                fmin = np.min(self.fu)
                w_raw = np.exp(-beta * (self.fu - fmin))
                # avoid zero weights
                if np.all(w_raw == 0):
                    w_raw = np.ones_like(w_raw)
                w = w_raw / (np.sum(w_raw) + 1e-16)  # normalized weights
        
                # vectorized pairwise distances: shape (K,K)
                # compute differences: for force, we need (u_j - u_i)
                U = self.u  # (K,dim)
                # pairwise diff: shape (K,K,dim) computed via broadcasting
                diffs = U[np.newaxis, :, :] - U[:, np.newaxis, :]  # diffs[i,j,:] = u_j - u_i
                # pairwise distances r_ij
                r = np.linalg.norm(diffs, axis=2)  # (K,K)
        
                # optionally limit neighbors using KDTree
                # but keep simple: use r_cut or neighbor_limit
                if (r_cut is not None) and (r_cut > 0):
                    mask_r = (r <= r_cut)
                else:
                    mask_r = np.ones_like(r, dtype=bool)
        
                # compute kernel G and derivative dGdr on r
                G, dGdr = kernel_G_and_dG(r, eps)  # both (K,K)
                # zero out self-interaction
                np.fill_diagonal(G, 0.0)
                np.fill_diagonal(dGdr, 0.0)
                # zero out beyond r_cut if requested
                G = G * mask_r
                dGdr = dGdr * mask_r
        
                # optionally limit by nearest neighbors per row (approx)
                if neighbor_limit is not None and neighbor_limit < K:
                    # for each i, mask out all but neighbor_limit largest |G| contributions
                    # compute absolute contributions: |w_j * G_ij|
                    abs_contrib = np.abs(w[np.newaxis, :] * G)
                    # for each row i find cutoff threshold
                    for i in range(K):
                        if neighbor_limit < K:
                            # indices of top contributors
                            # note: if many zeros, this still works.
                            topk = np.argpartition(-abs_contrib[i], min(neighbor_limit, K-1))[:neighbor_limit]
                            mask = np.zeros(K, dtype=bool)
                            mask[topk] = True
                            # keep zeros and topk
                            G[i, ~mask] = 0.0
                            dGdr[i, ~mask] = 0.0
        
                # compute forces Fi for each particle: Fi = sum_j w_j * (u_j - u_i) * (dGdr_ij / r_ij)
                # handle r==0 avoiding division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor = np.where(r > 0, dGdr / r, 0.0)  # (K,K)
                # weight by w_j
                weighted = (w[np.newaxis, :] * factor)[:, :, np.newaxis]  # shape (K,K,1)
                # element-wise multiply by diffs and sum over j
                forces = np.sum(weighted * diffs, axis=1)  # shape (K,dim)  (sum over j)
        
                # normalize force magnitudes to avoid explosion
                norms = np.linalg.norm(forces, axis=1)
                # avoid zero division
                norms_safe = np.where(norms == 0, 1.0, norms)
                forces_normalized = forces / norms_safe[:, np.newaxis]
        
                # directional sign: attraction toward good points already encoded via w; multiply by norms to scale
                # step update: x <- x + step * norms * normalized_force
                # we use scaled update: delta = step * (norms)[:,None] * normalized_force
                deltas = step * norms[:, np.newaxis] * forces_normalized
        
                # update positions
                new_u = self.u + deltas
        
                # enforce bounds if provided
                if len(self.lb) != 0 and len(self.ub) != 0:
                    lb_arr = np.array(self.lb, dtype=float)
                    ub_arr = np.array(self.ub, dtype=float)
                    new_u = np.minimum(np.maximum(new_u, lb_arr[np.newaxis, :]), ub_arr[np.newaxis, :])
        
                # evaluate new points
                new_fu = np.zeros_like(self.fu)
                for i in range(K):
                    new_fu[i] = self.get_f(new_u[i])
        
                # replace particle set
                self.u = new_u
                self.fu = new_fu
        
                # update best
                epoch_best_idx = np.argmin(self.fu)
                if self.fu[epoch_best_idx] < best_f:
                    best_f = float(self.fu[epoch_best_idx])
                    best_x = self.u[epoch_best_idx].copy()
        
                # optionally replace worst points with samples around best points (resampling/sift)
                if replace_worst_frac > 0:
                    n_replace = int(np.ceil(replace_worst_frac * K))
                    if n_replace >= 1:
                        # find worst indices
                        worst_idx = np.argsort(-self.fu)[:n_replace]
                        # sample new points around best_x with Gaussian perturbation proportional to eps
                        noise_scale = max(eps, 1e-3)
                        for j, wi in enumerate(worst_idx):
                            self.u[wi] = best_x + np.random.normal(scale=noise_scale, size=dim)
                            # clip
                            if len(self.lb) != 0 and len(self.ub) != 0:
                                self.u[wi] = np.minimum(np.maximum(self.u[wi], lb_arr), ub_arr)
                            self.fu[wi] = self.get_f(self.u[wi])
        
                # diagnostics
                if verbose and (epoch % max(1, max_epochs//10) == 0):
                    print(f"[potential] epoch {epoch}/{max_epochs}, best_f = {best_f:.6e}, "
                          f"mean_f = {np.mean(self.fu):.6e}, eps = {eps:.3e}, beta = {beta:.3f}")
        
                # anneal eps and beta
                eps = max(eps * eps_anneal_rate, eps_final)
                beta = min(beta * beta_anneal_rate, beta_final)
        
            # optional local refine
            if local_refine and (sp_minimize is not None):
                try:
                    res = sp_minimize(lambda x: float(self.get_f(x)), best_x, method='Nelder-Mead',
                                       options={'maxiter': 200, 'xatol': self.tol})
                    if res.success and res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = res.x.copy()
                except Exception:
                    pass
        
            # final set best into class for consistency
            self.best = getattr(self, 'best', {})
            self.best['x'] = best_x
            self.best['f'] = best_f
        
            return (best_f, best_x)

    def minimize_vidyasagar(self, alpha=0.01, beta=0.9, nesterov=False,
                        grad_eps=1e-6, maxIter=None, verbose=False):
            """
            Vidyasagar (unified momentum) style minimizer.
            Uses a finite-difference gradient approximation when an analytic gradient is unavailable.
        
            Parameters
            ----------
            alpha : float
                Step size (learning rate).
            beta : float
                Momentum coefficient (0 <= beta < 1).
            nesterov : bool
                If True, use Nesterov-style lookahead gradient evaluation.
            grad_eps : float
                Finite-difference epsilon for gradient approximation.
            maxIter : int or None
                Maximum iterations (if None, uses self.maxIter).
            verbose : bool
                If True, prints progress every 50 iterations.
        
            Returns
            -------
            (f_best, x_best)
                Tuple with best function value found and corresponding x.
            """
            if maxIter is None:
                maxIter = self.maxIter
        
            # Initialize
            x = np.array(self.X0, dtype=float).copy()
            v = np.zeros_like(x)
            f_x = self.get_f(x)
        
            f_best = f_x
            x_best = x.copy()
        
            # helper: finite-difference gradient (central)
            def approx_grad(z, eps=grad_eps):
                g = np.zeros(self.dim, dtype=float)
                # use central differences
                for i in range(self.dim):
                    ei = np.zeros(self.dim, dtype=float)
                    ei[i] = 1.0
                    zp = z + eps * ei
                    zm = z - eps * ei
                    fp = self.get_f(zp)
                    fm = self.get_f(zm)
                    g[i] = (fp - fm) / (2.0 * eps)
                return g
        
            t = 0
            while t < maxIter:
                if nesterov:
                    # lookahead position
                    x_look = x + beta * v
                    grad = approx_grad(x_look)
                else:
                    grad = approx_grad(x)
        
                grad_norm = np.linalg.norm(grad)
                if grad_norm < self.tol:
                    if verbose:
                        print(f"[vidyasagar] stopping: grad_norm {grad_norm:.3e} < tol {self.tol}")
                    break
        
                # velocity and position update (heavy-ball / unified)
                v = beta * v - alpha * grad
                x = x + v
        
                # enforce bounds if provided
                if len(self.lb) != 0 and len(self.ub) != 0:
                    x = np.minimum(np.maximum(x, np.array(self.lb, dtype=float)), np.array(self.ub, dtype=float))
        
                # evaluate
                f_x = self.get_f(x)
                if f_x < f_best:
                    f_best = f_x
                    x_best = x.copy()
        
                if verbose and (t % 50 == 0):
                    print(f"[vidyasagar] iter {t:4d} f = {f_x:.6e} grad_norm = {grad_norm:.3e}")
        
                t += 1
        
            # final evaluation (in case last iterate was best)
            return (f_best, x_best)

    
    def minimize_potential_second_order(self,
                                   kernel='inv_power',
                                   kernel_param=2.0,
                                   eps0=1.0,
                                   eps_final=1e-4,
                                   eps_anneal_rate=0.90,
                                   beta0=1.0,
                                   beta_final=50.0,
                                   beta_anneal_rate=1.08,
                                   step=0.2,
                                   max_epochs=150,
                                   replace_worst_frac=0.2,
                                   local_newton_steps=10,
                                   grad_eps=1e-6,
                                   hess_eps=1e-4,
                                   damp_init=1e-3,
                                   damp_increase=10.0,
                                   damp_reduce=0.5,
                                   accept_tol=1e-6,
                                   use_potential_hessian=False,
                                   bounds=True,
                                   verbose=False):
            """
            Potential-based global search with local second-order (damped-Newton) refinement.
            This method combines the Kaplinskii-style potential field movement with a
            Levenberg-Marquardt / damped-Newton update on promising points.
        
            Parameters
            ----------
            kernel, kernel_param, eps0, eps_final, eps_anneal_rate, beta0, beta_final, beta_anneal_rate:
                Same meanings as minimize_potential_nonlocal (potential construction + annealing).
            step : float
                global movement step multiplier for potential-driven move.
            max_epochs : int
                outer number of epochs (global potential updates).
            replace_worst_frac : float
                fraction of worst particles to resample around the best ones each epoch.
            local_newton_steps : int
                number of local Newton refinement iterations to attempt on top candidates per epoch.
            grad_eps : float
                finite-difference epsilon for gradient approximation.
            hess_eps : float
                finite-difference epsilon used to approximate Hessian (step for differences).
            damp_init : float
                initial damping (lambda) for Levenberg-Marquardt regularization.
            damp_increase / damp_reduce : float
                factors to adjust damping when steps fail or succeed.
            accept_tol : float
                required improvement threshold to accept a Newton step.
            use_potential_hessian : bool
                if True, attempt to approximate Hessian of the potential field; otherwise, approximate Hessian of the objective.
            bounds : bool
                enforce self.lb/self.ub if present.
            verbose : bool
                print diagnostics.
        
            Returns
            -------
            (best_f, best_x)
            """
            import numpy as _np
            from math import isfinite
        
            # --- helper finite-difference gradient (central) ---
            def approx_grad(x, eps=grad_eps):
                x = _np.asarray(x, dtype=float)
                g = _np.zeros(self.dim, dtype=float)
                for i in range(self.dim):
                    ei = _np.zeros(self.dim, dtype=float)
                    ei[i] = 1.0
                    f_plus = self.get_f(x + eps * ei)
                    f_minus = self.get_f(x - eps * ei)
                    g[i] = (f_plus - f_minus) / (2.0 * eps)
                return g
        
            # --- helper finite-difference Hessian (symmetric) ---
            def approx_hessian(x, eps=hess_eps):
                x = _np.asarray(x, dtype=float)
                H = _np.zeros((self.dim, self.dim), dtype=float)
                # diagonal via second differences
                f0 = self.get_f(x)
                for i in range(self.dim):
                    ei = _np.zeros(self.dim, dtype=float); ei[i] = 1.0
                    f_plus = self.get_f(x + eps * ei)
                    f_minus = self.get_f(x - eps * ei)
                    H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (eps * eps)
                # off-diagonals via mixed second differences
                for i in range(self.dim):
                    for j in range(i+1, self.dim):
                        ei = _np.zeros(self.dim, dtype=float); ei[i] = 1.0
                        ej = _np.zeros(self.dim, dtype=float); ej[j] = 1.0
                        f_pp = self.get_f(x + eps*ei + eps*ej)
                        f_pm = self.get_f(x + eps*ei - eps*ej)
                        f_mp = self.get_f(x - eps*ei + eps*ej)
                        f_mm = self.get_f(x - eps*ei - eps*ej)
                        H_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
                        H[i, j] = H_ij
                        H[j, i] = H_ij
                return H
        
            # --- potential helpers copied/adapted from your potential method style ---
            def kernel_G_and_dG(r, eps):
                if kernel == 'inv_power':
                    p = float(kernel_param)
                    rp = _np.power(r + 0.0, p)
                    denom = rp + eps
                    G = 1.0 / denom
                    with _np.errstate(divide='ignore', invalid='ignore'):
                        dG = - p * _np.power(r, max(p-1, 0.0)) / (denom * denom)
                    dG = _np.where(r == 0.0, 0.0, dG)
                    return G, dG
                elif kernel == 'inv_multiquadric':
                    denom = _np.sqrt(r*r + eps)
                    G = 1.0 / denom
                    dG = - r / _np.power(r*r + eps, 1.5)
                    dG = _np.where(r == 0.0, 0.0, dG)
                    return G, dG
                elif kernel == 'gaussian':
                    sigma = float(kernel_param)
                    G = _np.exp(-0.5 * (r / sigma)**2)
                    dG = (-r / (sigma**2)) * G
                    dG = _np.where(r == 0.0, 0.0, dG)
                    return G, dG
                else:
                    raise ValueError("Unknown kernel: " + str(kernel))
        
            # --- prepare population ---
            self.u = _np.asarray(self.u, dtype=float)
            if self.u.ndim == 1:
                self.u = self.u[_np.newaxis, :]
            K = self.u.shape[0]
            dim = self.dim
        
            if not hasattr(self, 'fu') or len(self.fu) != K:
                self.fu = _np.zeros(K)
                for i in range(K):
                    self.fu[i] = self.get_f(self.u[i])
        
            best_idx = _np.argmin(self.fu)
            best_x = self.u[best_idx].copy()
            best_f = float(self.fu[best_idx])
        
            eps = float(eps0)
            beta = float(beta0)
            lb_arr = _np.array(self.lb, dtype=float) if len(self.lb) != 0 else None
            ub_arr = _np.array(self.ub, dtype=float) if len(self.ub) != 0 else None
        
            for epoch in range(max_epochs):
                # weights from fu
                fmin = _np.min(self.fu)
                w_raw = _np.exp(-beta * (self.fu - fmin))
                if _np.all(w_raw == 0):
                    w_raw = _np.ones_like(w_raw)
                w = w_raw / (_np.sum(w_raw) + 1e-16)
        
                U = self.u  # (K, dim)
                diffs = U[_np.newaxis, :, :] - U[:, _np.newaxis, :]  # (K,K,dim) diffs[i,j]=u_j-u_i
                r = _np.linalg.norm(diffs, axis=2)
                G, dGdr = kernel_G_and_dG(r, eps)
                _np.fill_diagonal(G, 0.0); _np.fill_diagonal(dGdr, 0.0)
                with _np.errstate(divide='ignore', invalid='ignore'):
                    factor = _np.where(r > 0, dGdr / r, 0.0)
                weighted = (w[_np.newaxis, :] * factor)[:, :, _np.newaxis]
                forces = _np.sum(weighted * diffs, axis=1)  # (K, dim)
        
                norms = _np.linalg.norm(forces, axis=1)
                norms_safe = _np.where(norms == 0, 1.0, norms)
                forces_normalized = forces / norms_safe[:, _np.newaxis]
                deltas = step * norms[:, _np.newaxis] * forces_normalized
                # global potential move
                new_u = self.u + deltas
        
                # bounds
                if bounds and (lb_arr is not None and ub_arr is not None):
                    new_u = _np.minimum(_np.maximum(new_u, lb_arr[_np.newaxis, :]), ub_arr[_np.newaxis, :])
        
                # evaluate new candidates
                new_fu = _np.zeros_like(self.fu)
                for i in range(K):
                    new_fu[i] = self.get_f(new_u[i])
        
                # update population
                self.u = new_u
                self.fu = new_fu
        
                # update best
                epoch_best_idx = _np.argmin(self.fu)
                if self.fu[epoch_best_idx] < best_f:
                    best_f = float(self.fu[epoch_best_idx])
                    best_x = self.u[epoch_best_idx].copy()
        
                # --- local second-order refinement on top candidates ---
                # choose top-N (here top 3 or K small) for Newton refinement
                n_refine = min(3, max(1, K // 10))
                top_idx = _np.argsort(self.fu)[:n_refine]
                for idx in top_idx:
                    x_cur = self.u[idx].copy()
                    f_cur = float(self.fu[idx])
                    lam = float(damp_init)
                    # attempt several Newton iterations
                    for ln in range(local_newton_steps):
                        # choose which function to approximate Hessian on
                        if use_potential_hessian:
                            # approximate gradient of potential via weighted kernel grads
                            # compute gradient of potential at x_cur using current population
                            # gradPhi = sum_j w_j * (x_cur - u_j) * (dGdr_? / r)
                            # Here we compute numerical gradient of objective as fallback (safer)
                            grad = approx_grad(x_cur, eps=grad_eps)
                        else:
                            grad = approx_grad(x_cur, eps=grad_eps)
                        gnorm = _np.linalg.norm(grad)
                        if gnorm < self.tol:
                            break
                        # approximate Hessian
                        H = approx_hessian(x_cur, eps=hess_eps)
                        # regularize Hessian (LM style)
                        # ensure symmetry and positive-definiteness by adding lam * I
                        H_reg = H.copy()
                        # add to diagonal
                        H_reg += lam * _np.eye(dim)
                        # try to solve H_reg * p = -g
                        try:
                            p = _np.linalg.solve(H_reg, -grad)
                        except Exception:
                            # fallback to gradient step direction
                            p = -grad * (1.0 / (lam + 1e-8))
                        # line search backtracking on step size alpha
                        alpha = 1.0
                        f_new = None
                        for bs in range(10):
                            x_trial = x_cur + alpha * p
                            # bounds
                            if bounds and (lb_arr is not None and ub_arr is not None):
                                x_trial = _np.minimum(_np.maximum(x_trial, lb_arr), ub_arr)
                            f_trial = float(self.get_f(x_trial))
                            if f_trial < f_cur - accept_tol:
                                f_new = f_trial
                                break
                            alpha *= 0.5
                        if f_new is not None:
                            # accept step
                            x_cur = x_trial
                            f_cur = f_new
                            lam = max(lam * damp_reduce, 1e-16)
                        else:
                            # increase damping and try again
                            lam *= damp_increase
                        # end Newton iter
                    # end local newton iterations
                    # update population point if improved
                    if f_cur < self.fu[idx]:
                        self.u[idx] = x_cur.copy()
                        self.fu[idx] = f_cur
                        if f_cur < best_f:
                            best_f = float(f_cur)
                            best_x = x_cur.copy()
        
                # replace worst points with samples around best to keep exploration
                if replace_worst_frac > 0:
                    n_replace = int(_np.ceil(replace_worst_frac * K))
                    if n_replace >= 1:
                        worst_idx = _np.argsort(-self.fu)[:n_replace]
                        noise_scale = max(eps, 1e-3)
                        for wi in worst_idx:
                            self.u[wi] = best_x + _np.random.normal(scale=noise_scale, size=dim)
                            if bounds and (lb_arr is not None and ub_arr is not None):
                                self.u[wi] = _np.minimum(_np.maximum(self.u[wi], lb_arr), ub_arr)
                            self.fu[wi] = self.get_f(self.u[wi])
        
                if verbose and (epoch % max(1, max_epochs//10) == 0):
                    print(f"[pot2nd] epoch {epoch}/{max_epochs}, best_f={best_f:.6e}, mean_f={_np.mean(self.fu):.6e}, eps={eps:.3e}, beta={beta:.3f}")
        
                # anneal
                eps = max(eps * eps_anneal_rate, eps_final)
                beta = min(beta * beta_anneal_rate, beta_final)
        
            # final bookkeeping
            self.best = getattr(self, 'best', {})
            self.best['x'] = best_x
            self.best['f'] = best_f
            return (best_f, best_x)
 

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
    
                         # dimension (2D for easy visualization)
    lb = np.full(2, -5.0)        # lower bounds
    ub = np.full(2, 5.0)         # upper bounds

    X0 = np.array([0.8, 1.9])
    m = Minimization(of.ackley, X0, K=50, lb=lb, ub=ub)
    # initialize random population (Minimization constructor already does)
    best_f, best_x = m.minimize_potential_nonlocal(
        kernel='inv_power',
        kernel_param=2.0,
        eps0=2.0,
        eps_final=1e-4,
        eps_anneal_rate=0.92,
        beta0=0.5,
        beta_final=50.0,
        beta_anneal_rate=1.08,
        step=0.2,
        max_epochs=200,
        replace_worst_frac=0.25,
        local_refine=True,
        verbose=True)

    print("\nBest solution found by Kaplinskii potentia-based algorithm:")
    print("x =", best_x)
    print("f(x) =", best_f)
 
    m = Minimization(of.ackley, X0)
    res = m.minimize_vidyasagar(alpha=0.01, beta=0.95, nesterov=True, grad_eps=1e-6, maxIter=1000, verbose=True)
    print("Vidyasagar result:", res)
    
    m = Minimization(of.ackley, X0, K=60, lb=lb, ub=ub)
    best_f, best_x = m.minimize_potential_second_order(
            kernel='inv_power',
            kernel_param=2.0,
            eps0=2.0,
            eps_final=1e-6,
            eps_anneal_rate=0.92,
            beta0=0.5,
            beta_final=50.0,
            beta_anneal_rate=1.08,
            step=0.25,
            max_epochs=150,
            replace_worst_frac=0.25,
            local_newton_steps=6,
            grad_eps=1e-6,
            hess_eps=1e-3,
            damp_init=1e-3,
            verbose=True
        )
print("best by second-order potential method:", best_f, best_x)
