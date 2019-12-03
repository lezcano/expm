import math
import numpy as np
import torch

degs_dict = {"single": [1, 2, 4, 8, 15],
             "double": [1, 2, 4, 8, 15, 18]}

# TOOD(Lezcano): Single theta = 18 should be commented
thetas_dict = {"single": [1.192092800768788e-7,   # m_vals = 1
                          5.978858893805233e-04,  # m_vals = 2
                          5.116619363445086e-02,  # m_vals = 4
                          5.800524627688768e-01,  # m_vals = 8
                          2.3462], #2.217044394974720+e00 # deg 15
               "double": [2.220446049250313e-16,  # degs = 1
                          2.580956802971767e-08,  # degs = 2
                          3.397168839976962e-04,  # degs = 4
                          4.991228871115323e-02,  # degs = 8
                          6.7642e-01, #6.410835233041199e-01 # Extra error at deg 15 TODO: Why?
                          1.090863719290036e+00]# degs = 18
                }

def expm_taylor(A):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # Trivial case
    if A.shape == (1, 1):
        return torch.exp(A)

    if A.element_size() > 4:
        degs = degs_dict["double"]
        thetas = thetas_dict["double"]
    else:
        degs = degs_dict["single"]
        thetas = thetas_dict["single"]

    # No scale-square needed
    # This could be done marginally faster if iterated in reverse
    normA = torch.norm(A, 1).item()
    for deg, theta in zip(degs, thetas):
        if normA <= theta:
            return taylor_approx(A, deg)

    # Scale square
    s = int(np.ceil(np.log2(normA / thetas[-1])))
    # TOOD: What is this?
    # s = s - (t == 0.5); % adjust s if normA/theta(end) is a power of 2.
    A = A * (2**-s)
    X = taylor_approx(A, degs[-1])
    return torch.matrix_power(X, 2**s)


def taylor_approx(A, deg):
    I = torch.eye(A.shape[0], A.shape[1], dtype=A.dtype, device=A.device)

    if deg >= 2:
        A2 = A.mm(A)
    if deg > 8:
        A3 = A.mm(A2)
    if deg == 18:
        A6 = A3.mm(A3)

    if deg == 1:
        return I + A
    elif deg == 2:
        return I + A + .5 * A2
    elif deg == 4:
        return I + A + A2.mm(.5*I + A/6. + A2/24.)
    elif deg == 8:
        # TODO: Precompute
        SQRT = math.sqrt(177.)
        x3 = 2./3.
        a1 = 1./88.*(1.+SQRT)*x3
        a2 = 1./352.*(1.+SQRT)*x3
        u2 = 1/630*(857.-58.*SQRT)
        c0 = (-271.+29.*SQRT)/(315.*x3)
        c1 = (11.*(-1.+SQRT))/(1260.*x3)
        c2 = (11.*(-9.+SQRT))/(5040.*x3)
        c4 = -((-89.+SQRT)/(5040.*x3**2))
        # Matrix products
        A4 = A2*(a1*A + a2*A2)
        A8 = (x3*A2 + A4)*(c0*I + c1*A + c2*A2 + c4*A4)
        return I + A + u2*A2 + + A8
    elif deg == 15:
        b = [  1.,
              -0.1224230230553339932,
               0.3484665863364574400,
             -63.3171245588336972787,
              10.4080173523135410817,
              -0.1491449188999246000,
              -5.7923617070732609235,
               2.1163670172557469406,
               0.2381070373870987078,
              18.5714314142602603396,
               0.2684264296504340063,
              -0.0635231133561214716,
               0.4017568440673567886,
               0.0871216756605069225,
               0.0029455314402796828,
               0.0004018761610201036]
        y02 = A2.mm(b[15]*A2 + b[14]*A)
        y12 = (y02 + b[13]*A2 + b[12]*A).mm(y02 + b[11]*A2 + b[10]*I) + b[9]*y02
        return (y12 + b[8]*A2 + b[7]*A).mm(
                y12 + b[6]*y02 + b[5]*A) + \
                b[4]*y12 + b[3]*y02 +  b[2]*A2 + b[1]*A + b[0]*I
    elif deg == 18:
        b = torch.tensor(
        [[0.,
          0.,
          -10.96763960529620625935,
          -0.09043168323908105619,
          0.],

         [-0.10036558103014462001,
           0.39784974949964507614,
           1.68015813878906197182,
          -0.06764045190713819075,
           0.],

         [-0.00802924648241156960,
           1.36783778460411719922,
           0.05717798464788655127,
           0.06759613017704596460,
          -0.09233646193671185927],

         [-0.00089213849804572995,
           0.49828962252538267755,
          -0.00698210122488052084,
           0.02955525704293155274,
          -0.01693649390020817171],

         [ 0.,
          -0.00063789819459472330,
           0.00003349750170860705,
          -0.00001391802575160607,
          -0.00001400867981820361]],
        dtype=A.dtype, device=A.device)

        q = torch.stack([I, A, A2, A3, A6]).repeat(5, 1, 1, 1)
        # Transpose it because I put the matrix in the wrong row-column order...
        b = b.t().unsqueeze(-1).unsqueeze(-1).expand_as(q)
        q = (b * q).sum(dim=1)
        qaux = q[0].mm(q[4]) + q[3]
        return q[1] + (q[2] + qaux).mm(qaux)

        # The code above is an implementation of the following
        #q31 = a01*I + a11*A + a21*A2 + a31*A3
        #q61 = b01*I + b11*A + b21*A2 + b31*A3 + b61*A6
        #q62 = b02*I + b12*A + b22*A2 + b32*A3 + b62*A6
        #q63 = b03*I + b13*A + b23*A2 + b33*A3 + b63*A6
        #q64 = b04*I + b14*A + b24*A2 + b34*A3 + b64*A6
        #q91 = q31.mm(q64) + q63
        #return q61 + (q62 + q91).mm(q91)
