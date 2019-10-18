import torch
import numpy as np

from expm32 import expm32
from expm64 import expm64

DEBUG = False

def scale_square(X, exp):
    """
    Scale-squaring trick
    """
    norm = X.norm()
    if norm < .5:
        return exp(X)

    k = int(np.ceil(np.log2(float(norm)))) + 1
    B = X * (2.**-k)
    E = exp(B)
    for _ in range(k):
        E = torch.mm(E, E)
    return E


def taylor(X, n=20):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    coeff = [Id, X]
    for i in range(2, n):
        coeff.append(coeff[-1].mm(X) / i)
    return sum(coeff)

def expm_frechet(A, E, expm):
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=A.dtype, device=A.device, requires_grad=False)
    M[:n, :n] = A
    M[n:, n:] = A
    M[:n, n:] = E
    return expm(M)[:n, n:]

class expm_class(torch.autograd.Function):
    @staticmethod
    def _expm(A):
        if A.element_size() > 4:
            print("expm 64")
            expm = expm64
        else:
            print("expm 32")
            expm = expm32
        return expm(A)

    @staticmethod
    def _expm_frechet(A, E):
        if A.element_size() > 4:
            print("Frechet 64")
            expm = expm64
        else:
            print("Frechet 32")
            expm = expm32
        return expm_frechet(A, E, expm)

    @staticmethod
    def forward(ctx, A):
        B = expm_class._expm(A)
        ctx.save_for_backward(A, B)
        return B

    @staticmethod
    def backward(ctx, G):
        A, B = ctx.saved_tensors
        #if False:
        if torch.norm(A + A.t()) < 1e-7 and not DEBUG:
            print("Skew")
            # Optimise for skew-symmetric matrices
            def skew(X):
                return .5 * (X - X.t())
            grad = skew(B.t().matmul(G))
            out = B.matmul(expm_class._expm_frechet(-A, grad))
            # correct precission errors
            return skew(out)
        else:
            print("No Skew")
            Bt = B.t()
            grad = Bt.mm(G)
            dexp = expm_class._expm_frechet(A.t(), grad)
            # Compute (B^t)^{-1} * dexp
            return torch.solve(dexp, Bt).solution

expm = expm_class.apply



# Some very basic tests
A = torch.rand(5, 5, requires_grad=True)
A = torch.tensor([[0.2438, 0.3366, 0.6083, 0.4208, 0.2997],
                  [0.4911, 0.9196, 0.7790, 0.6629, 0.9682],
                  [0.4104, 0.3005, 0.1019, 0.9837, 0.2015],
                  [0.6454, 0.6973, 0.7667, 0.6931, 0.2697],
                  [0.9324, 0.4042, 0.8409, 0.7221, 0.7703]], requires_grad=True)
B = expm(A)
E = torch.tensor([[0.3891, 0.1785, 0.9886, 0.8972, 0.6448],
                  [0.9298, 0.3912, 0.9970, 0.2925, 0.2157],
                  [0.1791, 0.0150, 0.5072, 0.5781, 0.0153],
                  [0.2724, 0.5619, 0.8964, 0.2883, 0.5064],
                  [0.7171, 0.1772, 0.8602, 0.4367, 0.2689]])

def ret(A, B, E):
    return torch.autograd.grad([B], A, grad_outputs=(E,))[0]

# Test gradients with some random 32-bit vectors
print(ret(A, B, E))

# Test gradients with the 64 bit algorithm
A = A.double()
E = E.double()
B = expm(A)
print(ret(A, B, E))


# Test gradients with the 64 bit algorithm with a skew-symmetric matrix
# Rerun this same code with DEBUG = True (flag at the top of the script)
# ret1 should be the same with DEBUG = True and DEBUG = False
Aaux = .5 * (A - A.t())
E = E
B = expm(Aaux)
ret1 = ret(A, B, E)
print(ret1)

# Taylor + scale square should give us a quite decent approximation of the gradients
Aaux = .5 * (A - A.t())
B = scale_square(Aaux, taylor)
ret2 = ret(A, B, E)
print(ret2)
print(torch.norm(ret2-ret1).item())
