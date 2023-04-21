'''
'''

import torch
import torch.nn as nn

'''
'''
class SubSolver(nn.module):

    def __init__(self, max_iters, tol):
        super().__init__()

        self.max_iters = max_iters
        self.tol = tol

        return

    def _cubic(self, s, g, Hv, sigma):
        Hs = Hv(s)
        s_norm = torch.norm(s, 2)

        cubic_f = torch.dot(g, s) + 0.5*torch.dot(s, Hs) + (sigma/3)*s_norm.pow(3)
        cubic_g = g + Hs + (sigma*s_norm)*s

        return cubic_f, cubic_g
    
    def _cg_lanczos(self, g, Hv, sigma):

        g_norm = torch.norm(g, 2)
        tol = min(self.tol, self.tol*g_norm)

        d = torch.numel(g)
        K = min(d, self.max_iters)
        Q = torch.zeros((d, K))
        T = torch.zeros((K, K))
        q = g/g_norm

        for i in range(self.K-1):
            Q[:,i] = q
            v = Hv(q)
            T[i,i] = torch.dot(q, v)

            #Orthogonalize
            r = v - torch.matmul(Q[:,:i], torch.matmul(torch.transpose(Q[:,:i], 0, 1), v))

            b = torch.norm(r, 2)
            T[i,i+1] = b
            T[i+1,i] = b

            if b < tol:
                q = torch.zeros_like(q)
                break

            q = r/b

        #Compute last diagonal element
        T[i+1, i+1] = torch.dot(q, Hv(q))

        T = T[:i+2, :i+2]
        Q = Q[:, :i+2]

        '''Now that we have split up the setup and solve this logic needs to change'''
        if torch.norm(T) < tol and g_norm < 1e-16:
            return torch.zeros(d, device=device), 0

        gt = torch.matmul(torch.transpose(Q, 0, 1), g)

        self.sub_time += time.perf_counter() - tic

        return (gt, T, Q, tol)

    def forward(self, g, Hv, sigma):
        
        return self._cg_lanczos(g, Hv, sigma)