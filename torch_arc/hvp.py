'''
Hessian-vector product (hvp) operators using PyTorch AD. 
'''

import torch

'''Convert a list of tensors to a single 1D tensor'''
def list2vec(l):
    vec = [li.reshape(-1).data for li in l]

    return torch.cat(vec, 0)


'''Convert a 1D tensor to a list of tensors using template for the shapes'''
def vec2list(v, template):
    l = []
    start = 0
    for ti in template:
        l.append(v[start:start+torch.numel(ti)].reshape(ti.shape))
        start += torch.numel(ti)

    return l

'''
gradsH = torch.autograd.grad(loss(model(data[idx,...]), labels[idx,...]), model.parameters(), create_graph=True)
'''
def hvp(self, gradsH, param_groups):
    def Hv(v):
            v_list = vec2list(v, gradsH)

            hvp = torch.autograd.grad(gradsH, param_groups,
                                    grad_outputs=v_list, only_inputs=True,
                                    retain_graph=True)

            hvp_vec = list2vec(hvp)

            return hvp_vec

    return Hv