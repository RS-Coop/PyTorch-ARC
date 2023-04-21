'''
'''

import torch
from torch.optim import Optimizer

from torch_arc.subproblem_solver import SubSolver

class ARC(Optimizer):

    def __init__(self,
                 params,
                 sigma=1,
                 eta_1=0.1,
                 eta_2=0.9,
                 gamma_1=2,
                 gamma_2=2,
                 subprob_max_iters=50,
                 subprob_tol=1e-6):
        
        assert 0<eta_1 and eta_1<=eta_2 and eta_2<1
        assert 1<gamma_1 and gamma_1<=gamma_2
        
        defaults = dict(sigma=sigma,
                        eta_1=eta_1,
                        eta_2=eta_2,
                        gamma_1=gamma_1,
                        gamma_2=gamma_2)

        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("ARC doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']

        self.subsolver = SubSolver(subprob_max_iters, subprob_tol)

        return
    
    def _gather_flat_grad(self):

        views = []

        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)

        return torch.cat(views, 0)
    
    def _update(self, s):

        offset = 0

        for p in self._params:
            numel = p.numel()
            p.add_(s[offset:offset + numel].view_as(p))
            offset += numel

    @torch.no_grad()
    def step(self, closure):

        assert len(self.param_groups) == 1

        closure = torch.enable_grad()(closure)

        #
        group = self.param_groups[0]
        sigma = group['sigma']
        eta_1 = group['eta_1']
        eta_2 = group['eta_2']
        gamma_1 = group['gamma_1']
        gamma_2 = group['gamma_2']
        max_iters = group['max_iters']

        #compute loss and Hv operator
        loss, loss_fn, Hv = closure()

        #get gradients
        g = self._gather_flat_grad()

        #solve the sub-problem
        s, m = self.subsolver(g, Hv, sigma, max_iters)

        #update
        self._update(s)
        p = (loss_fn() - loss)/m

        #assume successful update
        success = True

        #bad update
        if p<eta_1:
            #unsuccessful update
            success = False

            #Undo update
            self._update(-s)

            #increase regularization
            group['sigma'] = gamma_1*sigma

        #great update
        elif p>eta_2:
            #decrease regularization
            group['sigma'] = max(sigma/gamma_2, 1e-16)

        #otherwise okay update, keep regularization

        return success