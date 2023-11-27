import math
exp = math.exp
sqrt = math.sqrt

def u_c(x,z):
    b = 0.12 * z
    lam = 1.35
    u0 = 10.0
    c0 = 10.0
    d0 = 2.0
    L0c = 5.2 * d0
    L0u = 6.4 * d0
    if z > L0c:
        decay = x * x / (b*b + 1E-6)    # longitudinal decay
        cm = c0 * 2.27 * sqrt(d0 / z)
        c = cm * exp(-decay / (lam*lam))
    else:
        r = (1.0 / L0c) * (L0c - z)
        if x > r:
            decay = (x - r) * (x - r) / (b*b + 1E-6)    # transversal decay
            c = c0 * exp(-decay / (lam*lam))
        else:
            c = c0

    if z > L0u:
        decay = x * x / (b*b + 1E-6)
        um = u0 * 2.58 * sqrt(d0 / z)
        u = um * exp(-decay)
    else:
        r = (1.0 / L0u) * (L0u - z)
        if x > r:
            decay = (x - r) * (x - r) / (b*b + 1E-6)
            u = u0 * exp(-decay)
        else:
            u = u0
    
    return u, c