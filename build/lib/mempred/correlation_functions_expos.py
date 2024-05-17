from sympy import nroots, roots, symbols, fraction, cancel, Poly, degree
import numpy as np
from sympy.abc import x
from scipy.optimize import least_squares


def roots_n_expos_sym(ai, taui, precision_roots=6, root_steps=1000):
    print('function too slow for n>3')
    n = len(ai)
    if len(taui) != n:
        print('not same number of amplitudes and decay times')
    da = {}
    dtau = {}
    for i in range(n):
        da["a{0}".format(i)] = symbols('a'+str(i))
        dtau["tau{0}".format(i)] = symbols('tau'+str(i))     
    w = symbols('w')

    p1 = 1
    p2 = 1
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(n):
        for j in range(n):
            p1 *= (w**2 + 1j * w /dtau['tau'+str(i)] - 1j * w /dtau['tau'+str(j)] + 1/(dtau['tau'+str(i)]*dtau['tau'+str(j)]))
            p_spec = 1
            for k in range(n):
                for l in range(n):
                    if k != i or l != j:
                        p_spec *= (w**2 + 1j * w /dtau['tau'+str(k)] - 1j * w /dtau['tau'+str(l)] + 1/(dtau['tau'+str(k)]*dtau['tau'+str(l)]))
            #print(p_spec)
            sum1 += w**2 * da['a'+str(i)] * da['a'+str(j)] * p_spec
            
        p2 *= (w**2 + 1/dtau['tau'+str(i)]**2) # could also be written as np.prod(w**2 + 1/taui**2) 
        p_sum = 1
        for m in range(n):
            if m != i:
                p_sum *= (w**2 + 1/dtau['tau'+str(m)]**2)

        sum2 += (da['a'+str(i)]/dtau['tau'+str(i)]) * p_sum
        sum3 += da['a'+str(i)] * p_sum

    

    integrand = cancel((-2*p1 * sum2) / (w**4 * p1 * (p2 - 2*sum3) + p2 * sum1)) # simplify ewpression and input eq. from report
    num, denom = fraction(integrand) # find nominator and denominator
    # num *= np.prod(taui**2)
    # denom *= np.prod(taui**2)
    #print(num)

    dan = {}
    dtaun = {}
    for i in range(n):
        dan["a{0}".format(i)] = ai[i]
        dtaun["tau{0}".format(i)] = taui[i]
    
    #print(sum1.expand().evalf(subs=dan|dtaun))

    ci = np.zeros(n+2)
    ki = np.zeros(n)

    for j in range(n):
        ki[j] = num.coeff(w, 2*j).evalf(subs=dan|dtaun)
    # 0, 2, 4, ... 2(n-1) are non-zero
    for j in range(1, n+3):
            ci[j-1] = denom.coeff(w, 2*j).evalf(subs=dan|dtaun)

    poly_solve = 0
    for i in range(len(ci)):
        poly_solve += ci[i] * x**(i)

    nroot = nroots(poly_solve, n=precision_roots, maxsteps=root_steps)

    return np.complex64(nroot), ki, ci


def roots_prw_harm(a, K):
    #roots of msd integral denominator for memory kernel of the form a*delta(t) and harmonic potential with U(x)/m = K/2 * x**2
    k1 = -2 * a
    c0 = K**2
    c1 = a**2 - 2 * K
    c2 = 1
    
    nroot = [-c1/2 + np.sqrt(c1**2/4 - c0 +0j), -c1/2 - np.sqrt(c1**2/4 - c0 +0j)] #nroots(c0 + x*(c2*x + c1), n=15)

    return np.complex128(nroot), np.array([k1]), np.array([c0, c1, c2])

def msd_prw_harm(a, K, B, t):
    r_i, k_i, c_i = roots_prw_harm(a, K)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * k_i[0]

    msd = B/c_i[-1] * summe
    return np.real(msd)




def roots_tri_oscexp_harm(a1, a2, a3, tau1, tau2, tau3, f1, f2, f3, K):
   a1 = np.array(a1, dtype=np.float128)
   a2= np.array(a2, dtype=np.float128)
   a3 = np.array(a3, dtype=np.float128)

   tau1 = np.array(tau1, dtype=np.float128)
   tau2= np.array(tau2, dtype=np.float128)
   tau3 = np.array(tau3, dtype=np.float128)

   f1 = np.array(f1, dtype=np.float128)
   f2= np.array(f2, dtype=np.float128)
   f3 = np.array(f3, dtype=np.float128)


   k1 = -4 * (1  + f1**2 * tau1**2) * (1  + f2**2 * tau2**2) * (1 + 
      f3**2 * tau3**2) * (a1 * tau1 * (1  + f2**2 * tau2**2) * (1 + 
         f3**2 * tau3**2)  + (1 + 
         f1**2 * tau1**2) * (a3 * tau3  + tau2 * (a2 + 
            a3 * f2**2 * tau2 * tau3  + a2 * f3**2 * tau3**2)))
   k2 = 8 * (a3 * (-tau2**2 + 
         f2**2 * tau2**4  + tau1**2 * (2 * f1**2 * tau2**2 * (-1 + 
               f2**2 * tau2**2)  - (1  + f2**2 * tau2**2)**2) + 
         f1**2 * tau1**4 * (f1**2 * tau2**2 * (-1  + f2**2 * tau2**2)  + (1 + 
               f2**2 * tau2**2)**2)) * tau3 * (1  + f3**2 * tau3**2) + 
      a2 * tau2 * (1  + f2**2 * tau2**2) * (-tau3**2 + 
         f3**2 * tau3**4  + tau1**2 * (2 * f1**2 * tau3**2 * (-1 + 
               f3**2 * tau3**2)  - (1  + f3**2 * tau3**2)**2) + 
         f1**2 * tau1**4 * (f1**2 * tau3**2 * (-1  + f3**2 * tau3**2)  + (1 + 
               f3**2 * tau3**2)**2)) + 
      a1 * tau1 * (1  + f1**2 * tau1**2) * (-tau3**2 + 
         f3**2 * tau3**4  + tau2**2 * (2 * f2**2 * tau3**2 * (-1 + 
               f3**2 * tau3**2)  - (1  + f3**2 * tau3**2)**2) + 
         f2**2 * tau2**4 * (f2**2 * tau3**2 * (-1  + f3**2 * tau3**2)  + (1 + 
               f3**2 * tau3**2)**2)))
   k3 = -4 * (a3 * (tau1**4 + 
         2 * tau1**2 * (2  + (-2 * f1**2  + f2**2) * tau1**2) * tau2**2  + (1 + 
            2 * (f1**2 - 2 * f2**2) * tau1**2  + (f1**4 + 4 * f1**2 * f2**2 + 
               f2**4) * tau1**4) * tau2**4) * tau3 * (1 + 
         f3**2 * tau3**2) + 
      a1 * tau1 * (1  + f1**2 * tau1**2) * (tau2**4 + 
         2 * tau2**2 * (2  + (-2 * f2**2  + f3**2) * tau2**2) * tau3**2  + (1 + 
            2 * (f2**2 - 2 * f3**2) * tau2**2  + (f2**4 + 4 * f2**2 * f3**2 + 
               f3**4) * tau2**4) * tau3**4) + 
      a2 * tau2 * (1  + f2**2 * tau2**2) * (tau3**4 + 
         2 * tau1**2 * tau3**2 * (2  + (f1**2 - 
               2 * f3**2) * tau3**2)  + tau1**4 * (1 + 
            2 * (-2 * f1**2  + f3**2) * tau3**2  + (f1**4 + 4 * f1**2 * f3**2 + 
               f3**4) * tau3**4)))
   k4 = 8 * tau1 * tau2 * tau3 * (a3 * tau1 * tau2 * (-tau2**2 + \
   tau1**2 * (-1  + (f1**2  + f2**2) * tau2**2)) * (1  + f3**2 * tau3**2) + 
      a2 * tau1 * (1 + 
         f2**2 * tau2**2) * tau3 * (-tau1**2  + (-1  + (f1**2 + 
               f3**2) * tau1**2) * tau3**2) + 
      a1 * (1  + f1**2 * tau1**2) * tau2 * tau3 * (-tau2**2  + (-1  + (f2**2 \
   + f3**2) * tau2**2) * tau3**2))
   k5 = -4 * (tau1 * tau2 * (a1 * (1  + f1**2 * tau1**2) * tau2**3 + 
         a2 * tau1**3 * (1  + f2**2 * tau2**2)) * tau3**4 + 
      a3 * tau1**4 * tau2**4 * tau3 * (1  + f3**2 * tau3**2))
   c0 = K**2 * (1  + f1**2 * tau1**2)**2 * (1  + f2**2 * tau2**2)**2 * (1 + 
      f3**2 * tau3**2)**2

   c1 = 4 * a1**2 * tau1**2 * (1  + f2**2 * tau2**2)**2 * (1  + f3**2 * tau3**2)**2 + K**2 * (2 * tau1**2 - 2 * f1**2 * tau1**4) * (1  + f2**2 * tau2**2)**2 * (1 + 
         f3**2 * tau3**2)**2 + 8 * a1 * tau1 * (1  + f1**2 * tau1**2) * (1  + f2**2 * tau2**2) * (1 + 
         f3**2 * tau3**2) * (a3 * tau3  + tau2 * (a2 + 
            a3 * f2**2 * tau2 * tau3  + a2 * f3**2 * tau3**2)) + 4 * (1  + f1**2 * tau1**2)**2 * (a3 * tau3  + tau2 * (a2 + 
            a3 * f2**2 * tau2 * tau3  + a2 * f3**2 * tau3**2))**2  + (K + 
         f1**2 * K * tau1**2)**2 * (-2 * (tau3 + 
            f2**2 * tau2**2 * tau3)**2 * (-1  + f3**2 * tau3**2) - 
      2 * (-1  + f2**2 * tau2**2) * (tau2  + f3**2 * tau2 * tau3**2)**2) - 2 * K * (a1 * tau1**2 * (-3  + f1**2 * tau1**2) * (1  + f2**2 * tau2**2)**2 * (1 + 
            f3**2 * tau3**2)**2  + (1 + 
            f1**2 * tau1**2)**2 * (a2 * (-3  + f2**2 * tau2**2) * (tau2 + 
               f3**2 * tau2 * tau3**2)**2  + (1 + 
               f2**2 * tau2**2)**2 * (a3 * tau3**2 * (-3 + 
                  f3**2 * tau3**2)  + (1  + f3**2 * tau3**2)**2)))
   c2 = K**2 * tau1**4 * (1  + f2**2 * tau2**2)**2 * (1 + f3**2 * tau3**2)**2  + (K + 
      f1**2 * K * tau1**2)**2 * ((1  + f2**2 * tau2**2)**2 * tau3**4 + 
      4 * tau2**2 * (-1  + f2**2 * tau2**2) * tau3**2 * (-1 + 
            f3**2 * tau3**2)  + tau2**4 * (1  + f3**2 * tau3**2)**2) - 16 * f1**2 * tau1**4 * (a3 * tau3  + tau2 * (a2 + 
            a3 * f2**2 * tau2 * tau3  + a2 * f3**2 * tau3**2))**2 + 4 * tau1**2 * (1 + 
         f1**2 * tau1**2) * (a3 * tau3  + tau2 * (a2 + 
            a3 * f2**2 * tau2 * tau3  + a2 * f3**2 * tau3**2))**2 + a1**2 * tau1**2 * (-8 * (tau3  + f2**2 * tau2**2 * tau3)**2 * (-1 + 
            f3**2 * tau3**2)  + tau1**2 * (1  + f2**2 * tau2**2)**2 * (1 + 
            f3**2 * tau3**2)**2 - 
      8 * (-1  + f2**2 * tau2**2) * (tau2  + f3**2 * tau2 * tau3**2)**2) + K**2 * (2 * tau1**2 - 
      2 * f1**2 * tau1**4) * (-2 * (tau3  + f2**2 * tau2**2 * tau3)**2 * (-1 + 
            f3**2 * tau3**2) - 
      2 * (-1  + f2**2 * tau2**2) * (tau2  + f3**2 * tau2 * tau3**2)**2) + 2 * a1 * tau1 * (-16 * a3 * f2**2 * (1 + 
            f1**2 * tau1**2) * tau2**4 * tau3 * (1  + f3**2 * tau3**2) + 
      4 * a3 * (1  + f1**2 * tau1**2) * tau2**2 * (1 + 
            f2**2 * tau2**2) * tau3 * (1  + f3**2 * tau3**2) + 
         a2 * tau2 * (-8 * (1  + f1**2 * tau1**2) * (1 + 
               f2**2 * tau2**2) * tau3**2 * (-1 + 
               f3**2 * tau3**2)  + tau1 * (-3 + 
               f1**2 * tau1**2) * tau2 * (-3  + f2**2 * tau2**2) * (1 + 
               f3**2 * tau3**2)**2)  + (1 + 
            f2**2 * tau2**2) * (4 * a3 * (1 + 
               f1**2 * tau1**2) * tau2**2 * tau3 * (1 + 
               f3**2 * tau3**2)  + tau1 * (-3  + f1**2 * tau1**2) * (1 + 
               f2**2 * tau2**2) * (a3 * tau3**2 * (-3  + f3**2 * tau3**2)  + (1 + 
               f3**2 * tau3**2)**2)))  + (1 + 
         f1**2 * tau1**2) * (4 * tau1**2 * (a3 * tau3  + tau2 * (a2 + 
               a3 * f2**2 * tau2 * tau3  + a2 * f3**2 * tau3**2))**2  + (1 + 
            f1**2 * tau1**2) * ((1  + f2**2 * tau2**2)**2 + 
         2 * (-3 * a3 * (1  + f2**2 * tau2**2)**2  + (f3  + f2**2 * f3 * tau2**2)**2 +
               a3**2 * (4 * tau2**2 - 4 * f2**2 * tau2**4)) * tau3**2  + (a3 + 
               f3**2)**2 * (1  + f2**2 * tau2**2)**2 * tau3**4 + 
         2 * a2 * tau2**2 * (-3 + 
               f2**2 * tau2**2) * (a3 * tau3**2 * (-3  + f3**2 * tau3**2)  + (1 + 
               f3**2 * tau3**2)**2) + 
            a2**2 * tau2**2 * (8 * tau3**2 - 
            8 * f3**2 * tau3**4  + (tau2 + 
               f3**2 * tau2 * tau3**2)**2))) - 2 * K * (-4 * f1**2 * tau1**4 * (a2 * (-3  + f2**2 * tau2**2) * (tau2 + 
               f3**2 * tau2 * tau3**2)**2  + (1 + 
               f2**2 * tau2**2)**2 * (a3 * tau3**2 * (-3 + 
                  f3**2 * tau3**2)  + (1 + 
               f3**2 * tau3**2)**2))  + tau1**2 * (1 + 
            f1**2 * tau1**2) * (a2 * (-3  + f2**2 * tau2**2) * (tau2 + 
               f3**2 * tau2 * tau3**2)**2  + (1 + 
               f2**2 * tau2**2)**2 * (a3 * tau3**2 * (-3 + 
                  f3**2 * tau3**2)  + (1  + f3**2 * tau3**2)**2)) + 
         a1 * tau1**2 * (-tau1**2 * (1  + f2**2 * tau2**2)**2 * (1 + 
               f3**2 * tau3**2)**2  + (-3 + 
               f1**2 * tau1**2) * (-2 * (tau3 + 
                  f2**2 * tau2**2 * tau3)**2 * (-1  + f3**2 * tau3**2) - 
            2 * (-1  + f2**2 * tau2**2) * (tau2 + 
                  f3**2 * tau2 * tau3**2)**2))  + (1 + 
            f1**2 * tau1**2) * ((1 + 
               f1**2 * tau1**2) * (2 * tau2**2  - (a2 + 2 * f2**2) * tau2**4 + 
            2 * tau3**2 + 
            2 * tau2**2 * (-3 * a2 - 3 * a3 + 
               2 * (f2**2 + 
                     f3**2)  + (f2**2 * (a2 + 3 * a3  + f2**2)  - (a2 + 
                     2 * f2**2) * f3**2) * tau2**2) * tau3**2  - (a3 + 
               2 * f3**2 - 
               2 * (-a3 * f2**2  + (3 * a2  + a3 - 2 * f2**2) * f3**2 + 
                     f3**4) * tau2**2  + (a3 * f2**4 + 
                  2 * f2**2 * (a2  + a3  + f2**2) * f3**2  + (a2 + 
               
                        2 * f2**2) * f3**4) * tau2**4) * tau3**4) + \
   tau1**2 * (a2 * (-3  + f2**2 * tau2**2) * (tau2 + 
                  f3**2 * tau2 * tau3**2)**2  + (1 + 
                  f2**2 * tau2**2)**2 * (a3 * tau3**2 * (-3 + 
                     f3**2 * tau3**2)  + (1  + f3**2 * tau3**2)**2))))
   c3 = K**2 * (2 * tau1**2 - 
      2 * f1**2 * tau1**4) * ((1  + f2**2 * tau2**2)**2 * tau3**4 + 
      4 * tau2**2 * (-1  + f2**2 * tau2**2) * tau3**2 * (-1 + 
            f3**2 * tau3**2)  + tau2**4 * (1  + f3**2 * tau3**2)**2) + 4 * tau1**4 * (a3 * tau3  + tau2 * (a2  + a3 * f2**2 * tau2 * tau3 + 
            a2 * f3**2 * tau3**2))**2 + 2 * (K  + f1**2 * K * tau1**2)**2 * tau2**2 * tau3**2 * (tau3**2 + \
   tau2**2 * (1  - (f2**2  + f3**2) * tau3**2)) + K**2 * tau1**4 * (-2 * (tau3  + f2**2 * tau2**2 * tau3)**2 * (-1 + 
            f3**2 * tau3**2) - 
      2 * (-1  + f2**2 * tau2**2) * (tau2  + f3**2 * tau2 * tau3**2)**2) - 4 * f1**2 * tau1**4 * ((1  + f2**2 * tau2**2)**2 + 
      2 * (-3 * a3 * (1  + f2**2 * tau2**2)**2  + (f3  + f2**2 * f3 * tau2**2)**2 + 
            a3**2 * (4 * tau2**2 - 4 * f2**2 * tau2**4)) * tau3**2  + (a3 + 
            f3**2)**2 * (1  + f2**2 * tau2**2)**2 * tau3**4 + 
      2 * a2 * tau2**2 * (-3 + 
            f2**2 * tau2**2) * (a3 * tau3**2 * (-3  + f3**2 * tau3**2)  + (1 + 
            f3**2 * tau3**2)**2) + 
         a2**2 * tau2**2 * (8 * tau3**2 - 
         8 * f3**2 * tau3**4  + (tau2 + 
            f3**2 * tau2 * tau3**2)**2))  + tau1**2 * (1 + 
         f1**2 * tau1**2) * ((1  + f2**2 * tau2**2)**2 + 
      2 * (-3 * a3 * (1  + f2**2 * tau2**2)**2  + (f3  + f2**2 * f3 * tau2**2)**2 + 
            a3**2 * (4 * tau2**2 - 4 * f2**2 * tau2**4)) * tau3**2  + (a3 + 
            f3**2)**2 * (1  + f2**2 * tau2**2)**2 * tau3**4 + 
      2 * a2 * tau2**2 * (-3 + 
            f2**2 * tau2**2) * (a3 * tau3**2 * (-3  + f3**2 * tau3**2)  + (1 + 
            f3**2 * tau3**2)**2) + 
         a2**2 * tau2**2 * (8 * tau3**2 - 
         8 * f3**2 * tau3**4  + (tau2  + f3**2 * tau2 * tau3**2)**2)) + a1**2 * tau1**2 * (4 * ((1  + f2**2 * tau2**2)**2 * tau3**4 + 
         4 * tau2**2 * (-1  + f2**2 * tau2**2) * tau3**2 * (-1 + 
               f3**2 * tau3**2)  + tau2**4 * (1 + 
               f3**2 * tau3**2)**2)  + tau1**2 * (-2 * (tau3 + 
               f2**2 * tau2**2 * tau3)**2 * (-1  + f3**2 * tau3**2) - 
         2 * (-1  + f2**2 * tau2**2) * (tau2 + 
               f3**2 * tau2 * tau3**2)**2))  + (1 + 
         f1**2 * tau1**2) * (2 * (1 + 
            f1**2 * tau1**2) * (tau2**2  - (a2 + 
               f2**2) * tau2**4  + tau3**2  + tau2**2 * (2 * (-3 * a2 - 3 * a3 + 
                  f2**2  + f3**2)  + (a2**2 + 3 * a2 * a3 + 2 * a3**2 + 2 * a2 * f2**2 + 
               6 * a3 * f2**2  + f2**4 - 
               2 * (a2  + f2**2) * f3**2) * tau2**2) * tau3**2  - (a3 + 
               f3**2  - (2 * a2**2  + (a3  + f3**2) * (a3 - 2 * f2**2  + f3**2) + 
               3 * a2 * (a3 + 2 * f3**2)) * tau2**2  + (a2  + a3  + f2**2 + 
                  f3**2) * (a3 * f2**2  + (a2 + 
                     f2**2) * f3**2) * tau2**4) * tau3**4)  + tau1**2 * ((1 + 
            f2**2 * tau2**2)**2 + 
         2 * (-3 * a3 * (1  + f2**2 * tau2**2)**2  + (f3  + f2**2 * f3 * tau2**2)**2 +
               a3**2 * (4 * tau2**2 - 4 * f2**2 * tau2**4)) * tau3**2  + (a3 + 
               f3**2)**2 * (1  + f2**2 * tau2**2)**2 * tau3**4 + 
         2 * a2 * tau2**2 * (-3 + 
               f2**2 * tau2**2) * (a3 * tau3**2 * (-3  + f3**2 * tau3**2)  + (1 + 
               f3**2 * tau3**2)**2) + 
            a2**2 * tau2**2 * (8 * tau3**2 - 
            8 * f3**2 * tau3**4  + (tau2 + 
               f3**2 * tau2 * tau3**2)**2))) + 2 * a1 * tau1 * (4 * a3 * (1  + f1**2 * tau1**2) * tau2**4 * tau3 * (1 + 
            f3**2 * tau3**2) - 
      4 * f2**2 * tau1 * (-3 + 
            f1**2 * tau1**2) * tau2**4 * (a3 * tau3**2 * (-3 + 
               f3**2 * tau3**2)  + (1  + f3**2 * tau3**2)**2)  + tau1 * (-3 + 
            f1**2 * tau1**2) * tau2**2 * (1 + 
            f2**2 * tau2**2) * (a3 * tau3**2 * (-3  + f3**2 * tau3**2)  + (1 + 
            f3**2 * tau3**2)**2) + 
         a2 * tau2 * (4 * (1  + f1**2 * tau1**2) * (1 + 
               f2**2 * tau2**2) * tau3**4  - tau1 * (-3 * tau2**3 + \
   tau1**2 * tau2 * (-3  + (f1**2  + f2**2) * tau2**2)) * (1 + 
               f3**2 * tau3**2)**2  + tau1 * (-3 + 
               f1**2 * tau1**2) * tau2 * (-3 + 
               f2**2 * tau2**2) * (2 * tau3**2 - 2 * f3**2 * tau3**4))  + (1 + 
            f2**2 * tau2**2) * (tau1 * (-3 + 
               f1**2 * tau1**2) * tau2**2 * (a3 * tau3**2 * (-3 + 
                  f3**2 * tau3**2)  + (1  + f3**2 * tau3**2)**2)  + tau1 * (1 +
               f2**2 * tau2**2) * (-6 * tau3**2 + 
            3 * (a3 + 
               2 * f3**2) * tau3**4  + tau1**2 * (-1  + (3 * a3 + 
                  2 * (f1  - f3) * (f1  + f3)) * tau3**2  - (a3 * f1**2  + (a3 + 
                     2 * f1**2) * f3**2  + f3**4) * tau3**4)))) - 2 * K * (tau1**2 * (1 + 
            f1**2 * tau1**2) * (2 * tau2**2  - (a2 + 2 * f2**2) * tau2**4 + 
         2 * tau3**2 + 
         2 * tau2**2 * (-3 * a2 - 3 * a3 + 
            2 * (f2**2 + 
                  f3**2)  + (f2**2 * (a2 + 3 * a3  + f2**2)  - (a2 + 
                  2 * f2**2) * f3**2) * tau2**2) * tau3**2  - (a3 + 2 * f3**2 - 
            2 * (-a3 * f2**2  + (3 * a2  + a3 - 2 * f2**2) * f3**2 + 
                  f3**4) * tau2**2  + (a3 * f2**4 + 
               2 * f2**2 * (a2  + a3  + f2**2) * f3**2  + (a2 + 
                  2 * f2**2) * f3**4) * tau2**4) * tau3**4)  + tau1**4 * (a2 *(-3  + f2**2 * tau2**2) * (tau2  + f3**2 * tau2 * tau3**2)**2  + (1 + 
               f2**2 * tau2**2)**2 * (a3 * tau3**2 * (-3 + 
                  f3**2 * tau3**2)  + (1  + f3**2 * tau3**2)**2)) + 
         a1 * tau1**2 * ((-3 + 
               f1**2 * tau1**2) * ((1  + f2**2 * tau2**2)**2 * tau3**4 + 
            4 * tau2**2 * (-1  + f2**2 * tau2**2) * tau3**2 * (-1 + 
                  f3**2 * tau3**2)  + tau2**4 * (1  + f3**2 * tau3**2)**2) + 
         2 * tau1**2 * ((tau3  + f2**2 * tau2**2 * tau3)**2 * (-1 + 
                  f3**2 * tau3**2)  + (-1  + f2**2 * tau2**2) * (tau2 + 
                  f3**2 * tau2 * tau3**2)**2))  + (1 + 
            f1**2 * tau1**2) * (tau1**2 * (2 * tau2**2  - (a2 + 
               2 * f2**2) * tau2**4 + 2 * tau3**2 + 
            2 * tau2**2 * (-3 * a2 - 3 * a3 + 
               2 * (f2**2 + 
                     f3**2)  + (f2**2 * (a2 + 3 * a3  + f2**2)  - (a2 + 
                     2 * f2**2) * f3**2) * tau2**2) * tau3**2  - (a3 + 
               2 * f3**2 - 
               2 * (-a3 * f2**2  + (3 * a2  + a3 - 2 * f2**2) * f3**2 + 
                     f3**4) * tau2**2  + (a3 * f2**4 + 
                  2 * f2**2 * (a2  + a3  + f2**2) * f3**2  + (a2 + 
                     2 * f2**2) * f3**4) * tau2**4) * tau3**4)  + (1 + 
               f1**2 * tau1**2) * (tau3**4  + tau2**2 * tau3**2 * (4  - (3 * a2 + 2 * a3 - 2 * f2**2 + 4 * f3**2) * tau3**2)  + tau2**4 * (1  - (2 * a2 + 
                  3 * a3 + 4 * f2**2 - 
                  2 * f3**2) * tau3**2  + (f2**2 * (a2 + 2 * a3 + 
                        f2**2)  + (2 * a2  + a3 + 4 * f2**2) * f3**2 + 
                     f3**4) * tau3**4))) + 
      4 * f1**2 * tau1**4 * (-2 * tau3**2  + (a3 + 
            2 * f3**2) * tau3**4  + tau2**4 * (a2 + 2 * f2**2 - 
            2 * (f2**2 * (a2 + 3 * a3  + f2**2)  - (a2 + 
                  2 * f2**2) * f3**2) * tau3**2  + (a3 * f2**4 + 
               2 * f2**2 * (a2  + a3  + f2**2) * f3**2  + (a2 + 
                  2 * f2**2) * f3**4) * tau3**4) + 
         2 * tau2**2 * (-1  + tau3**2 * (3 * a2 - 2 * (f2**2  + f3**2) - 
                  f3**2 * (3 * a2  + a3 - 2 * f2**2  + f3**2) * tau3**2 + 
                  a3 * (3  + f2**2 * tau3**2)))))
   c4 = (K  + f1**2 * K * tau1**2)**2 * tau2**4 * tau3**4 + 2 * tau1**2 * (1 + 
         f1**2 * tau1**2) * (tau2**2  - (a2 + 
            f2**2) * tau2**4  + tau3**2  + tau2**2 * (2 * (-3 * a2 - 3 * a3 + 
               f2**2  + f3**2)  + (a2**2 + 3 * a2 * a3 + 2 * a3**2 + 2 * a2 * f2**2 + 
            6 * a3 * f2**2  + f2**4 - 
            2 * (a2  + f2**2) * f3**2) * tau2**2) * tau3**2  - (a3 + 
            f3**2  - (2 * a2**2  + (a3  + f3**2) * (a3 - 2 * f2**2  + f3**2) + 
            3 * a2 * (a3 + 2 * f3**2)) * tau2**2  + (a2  + a3  + f2**2 + 
               f3**2) * (a3 * f2**2  + (a2  + f2**2) * f3**2) * tau2**4) * tau3**4) + K**2 * tau1**4 * ((1  + f2**2 * tau2**2)**2 * tau3**4 + 
      4 * tau2**2 * (-1  + f2**2 * tau2**2) * tau3**2 * (-1 + 
            f3**2 * tau3**2)  + tau2**4 * (1  + f3**2 * tau3**2)**2) + 4 * K**2 * tau1**2 * (-1 + 
         f1**2 * tau1**2) * (-tau2**2 * tau3**4  + tau2**4 * tau3**2 * (-1 \
   + (f2**2  + f3**2) * tau3**2)) + 8 * f1**2 * tau1**4 * (-tau3**2  + (a3 + 
            f3**2) * tau3**4  + tau2**4 * (a2 + 
            f2**2  - (a2**2 + 3 * a2 * a3 + 2 * a3**2 + 2 * a2 * f2**2 + 6 * a3 * f2**2 + 
               f2**4 - 2 * (a2  + f2**2) * f3**2) * tau3**2  + (a2  + a3  + f2**2 + 
               f3**2) * (a3 * f2**2  + (a2 + 
                  f2**2) * f3**2) * tau3**4)  + tau2**2 * (-1 + 
         2 * (3 * a2 + 3 * a3  - f2**2 - 
               f3**2) * tau3**2  - (2 * a2**2  + (a3  + f3**2) * (a3 - 2 * f2**2 + 
                  f3**2) + 3 * a2 * (a3 + 2 * f3**2)) * tau3**4)) + a1**2 * tau1**2 * (8 * tau2**2 * tau3**4 + 
      8 * tau2**4 * tau3**2 * (1  - (f2**2 + 
               f3**2) * tau3**2)  + tau1**2 * ((1 + 
               f2**2 * tau2**2)**2 * tau3**4 + 
         4 * tau2**2 * (-1  + f2**2 * tau2**2) * tau3**2 * (-1 + 
               f3**2 * tau3**2)  + tau2**4 * (1 + 
               f3**2 * tau3**2)**2))  + tau1**4 * ((1  + f2**2 * tau2**2)**2 + 
      2 * (-3 * a3 * (1  + f2**2 * tau2**2)**2  + (f3  + f2**2 * f3 * tau2**2)**2 + 
            a3**2 * (4 * tau2**2 - 4 * f2**2 * tau2**4)) * tau3**2  + (a3 + 
            f3**2)**2 * (1  + f2**2 * tau2**2)**2 * tau3**4 + 
      2 * a2 * tau2**2 * (-3 + 
            f2**2 * tau2**2) * (a3 * tau3**2 * (-3  + f3**2 * tau3**2)  + (1 + 
            f3**2 * tau3**2)**2) + 
         a2**2 * tau2**2 * (8 * tau3**2 - 
         8 * f3**2 * tau3**4  + (tau2  + f3**2 * tau2 * tau3**2)**2)) + 2 * a1 * tau1 * (3 * tau1 * (-tau3**4  + tau2**2 * tau3**2 * (-4  + (3 *a2 + 2 * a3 - 2 * f2**2 + 4 * f3**2) * tau3**2)  + tau2**4 * (-1  + (2 * a2 + 
               3 * a3 + 4 * f2**2 - 
               2 * f3**2) * tau3**2  - (f2**2 * (a2 + 2 * a3  + f2**2)  + (2 * a2 + 
                     a3 + 4 * f2**2) * f3**2 + 
                  f3**4) * tau3**4))  + tau1**3 * (-2 * tau3**2  + (a3 + 
               f1**2 + 2 * f3**2) * tau3**4  + tau2**2 * (-2 + 
            2 * (3 * a2 + 3 * a3 + 2 * f1**2 - 2 * f2**2 - 
               2 * f3**2) * tau3**2  - (3 * a2 * f1**2 + 2 * a3 * f1**2 - 
               2 * a3 * f2**2 - 2 * f1**2 * f2**2 + 
               2 * (3 * a2  + a3 + 2 * f1**2 - 2 * f2**2) * f3**2 + 
               2 * f3**4) * tau3**4)  + tau2**4 * (a2  + f1**2 + 
            2 * f2**2  - (2 * a2 * f1**2 + 3 * a3 * f1**2 + 2 * a2 * f2**2 + 6 * a3 * f2**2 + 
               4 * f1**2 * f2**2 + 2 * f2**4 - 
               2 * (a2  + f1**2 + 2 * f2**2) * f3**2) * tau3**2  + ((a2 + 
                  2 * a3) * f1**2 * f2**2  + (a3 + 
                     f1**2) * f2**4  + ((2 * a2  + a3) * f1**2 + 
                  2 * (a2  + a3 + 2 * f1**2) * f2**2 + 2 * f2**4) * f3**2  + (a2 + 
                     f1**2 + 2 * f2**2) * f3**4) * tau3**4))) + 2 * K * (-2 * tau2**2 * tau3**4  + tau2**4 * tau3**2 * (-2  + (a2  + a3 + 
            2 * (f2**2  + f3**2)) * tau3**2) - 
      2 * tau1**2 * (tau2**4  + tau2**2 * (4  - (3 * a1 + 2 * a2 + 3 * a3 - 
               2 * (f1**2 - 2 * f2**2 + 
                     f3**2)) * tau2**2) * tau3**2  + (1  + tau2**2 * (-3 * a1 \
   - 3 * a2 + 2 * (-a3  + f1**2  + f2**2 - 2 * f3**2)  - (a2 + 
                     a3) * f1**2 * tau2**2  + (f2**2 * (3 * a1  + a2 + 2 * a3 - 
                     2 * f1**2  + f2**2)  + (3 * a1 + 2 * a2  + a3 - 2 * f1**2 + 
                     4 * f2**2) * f3**2 + 
                     f3**4) * tau2**2)) * tau3**4)  + tau1**4 * (-2 * tau3**2  + (a1  + a3 + 2 * (f1**2  + f3**2)) * tau3**4 + 
         2 * tau2**2 * (-1  + (2 * a1 + 3 * a2 + 3 * a3 + 4 * f1**2 - 
               2 * (f2**2  + f3**2)) * tau3**2  - (a1 * f1**2 + 3 * a2 * f1**2 + 
               2 * a3 * f1**2  + f1**4  - a1 * f2**2  - a3 * f2**2 - 
               2 * f1**2 * f2**2  + (2 * a1 + 3 * a2  + a3 + 4 * f1**2 - 
                  2 * f2**2) * f3**2  + f3**4) * tau3**4)  + tau2**4 * (a1 + 
               a2 + 2 * (f1**2  + f2**2) - 
            2 * (a1 * f1**2 + 2 * a2 * f1**2 + 3 * a3 * f1**2  + f1**4 + 2 * a1 * f2**2 + 
                  a2 * f2**2 + 3 * a3 * f2**2 + 4 * f1**2 * f2**2 + 
                  f2**4  - (a1  + a2 + 
                  2 * (f1**2  + f2**2)) * f3**2) * tau3**2  + (a2 * f1**4 + 
                  a3 * f1**4 + 2 * a1 * f1**2 * f2**2 + 2 * a2 * f1**2 * f2**2 + 
               4 * a3 * f1**2 * f2**2 + 2 * f1**4 * f2**2  + a1 * f2**4  + a3 * f2**4 + 
               2 * f1**2 * f2**4 + 
               2 * (f1**2 * (a1 + 2 * a2  + a3  + f1**2)  + (2 * a1  + a2  + a3 + 
                     4 * f1**2) * f2**2  + f2**4) * f3**2  + (a1  + a2 + 
                  2 * (f1**2  + f2**2)) * f3**4) * tau3**4)))  + (1 + 
         f1**2 * tau1**2) * (2 * tau1**2 * (tau2**2  - (a2 + 
               f2**2) * tau2**4  + tau3**2  + tau2**2 * (2 * (-3 * a2 - 3 * a3 + 
                  f2**2  + f3**2)  + (a2**2 + 3 * a2 * a3 + 2 * a3**2 + 2 * a2 * f2**2 + 
               6 * a3 * f2**2  + f2**4 - 
               2 * (a2  + f2**2) * f3**2) * tau2**2) * tau3**2  - (a3 + 
               f3**2  - (2 * a2**2  + (a3  + f3**2) * (a3 - 2 * f2**2  + f3**2) + 
               3 * a2 * (a3 + 2 * f3**2)) * tau2**2  + (a2  + a3  + f2**2 + 
                  f3**2) * (a3 * f2**2  + (a2 + 
                     f2**2) * f3**2) * tau2**4) * tau3**4)  + (1 + 
            f1**2 * tau1**2) * (tau3**4 + 
         2 * tau2**2 * tau3**2 * (2  + (-3 * a2 - 2 * a3  + f2**2 - 
               2 * f3**2) * tau3**2)  + tau2**4 * (1 + 
            2 * (-2 * a2 - 3 * a3 - 2 * f2**2  + f3**2) * tau3**2  + (a2**2 + 
                  a3**2 + 4 * a3 * f2**2  + f2**4 + 2 * (a3 + 2 * f2**2) * f3**2  + f3**4 + 
               2 * a2 * (a3  + f2**2 + 2 * f3**2)) * tau3**4)))
   c5 = 2 * (tau2**2 * tau3**4  + tau2**4 * tau3**2 * (1  - (a2  + a3 + 
            f2**2  + f3**2  + K) * tau3**2)  + tau1**2 * (tau3**4 + 
         2 * tau2**2 * tau3**2 * (2  + (-3 * a1 - 3 * a2 - 2 * a3  + f1**2 + 
               f2**2 - 2 * (f3**2  + K)) * tau3**2)  + tau2**4 * (1 + 
            2 * (-3 * a1 - 2 * a2 - 3 * a3  + f1**2 - 2 * f2**2  + f3**2 - 
               2 * K) * tau3**2  + (2 * a1**2  + a2**2  + a3**2 - 2 * a3 * f1**2 + 
               4 * a3 * f2**2 - 2 * f1**2 * f2**2  + f2**4 + 2 * a3 * f3**2 - 
               2 * f1**2 * f3**2 + 4 * f2**2 * f3**2  + f3**4 + 
               2 * (a3  - f1**2 + 2 * (f2**2  + f3**2)) * K  + K**2 + 
               2 * a2 * (a3  - f1**2  + f2**2 + 2 * f3**2  + K) + 
               3 * a1 * (a2  + a3 + 2 * (f2**2  + f3**2) + 
                  K)) * tau3**4))  - tau1**4 * (-tau3**2  + (a1  + a3 + 
            f1**2  + f3**2  + K) * tau3**4  - tau2**2 * (1 + 
            2 * (-2 * a1 - 3 * a2 - 3 * a3 - 2 * f1**2  + f2**2  + f3**2 - 
               2 * K) * tau3**2  + (a1**2 + 2 * a2**2  + a3**2 + 4 * a3 * f1**2 + 
               f1**4 - 2 * a3 * f2**2 - 2 * f1**2 * f2**2 + 2 * a3 * f3**2 + 
               4 * f1**2 * f3**2 - 2 * f2**2 * f3**2  + f3**4 + 
               2 * (a3 + 2 * f1**2  - f2**2 + 2 * f3**2) * K  + K**2 + 
               3 * a2 * (a3 + 2 * (f1**2  + f3**2)  + K) + 
               a1 * (3 * a2 + 
                  2 * (a3  + f1**2  - f2**2 + 2 * f3**2 + 
                     K))) * tau3**4)  + tau2**4 * (a1  + a2  + f1**2 + 
            f2**2  + K  - (a1**2  + a2**2 + 3 * a2 * a3 + 2 * a3**2 + 6 * a3 * f1**2 + 
               f1**4 + 6 * a3 * f2**2 + 4 * f1**2 * f2**2  + f2**4 - 2 * f1**2 * f3**2 - 
               2 * f2**2 * f3**2 + 3 * a3 * K + 4 * (f1**2  + f2**2) * K - 2 * f3**2 * K + 
               K**2 + 2 * a2 * (2 * f1**2  + f2**2  - f3**2  + K) + 
               a1 * (2 * a2 + 3 * a3 + 
                  2 * (f1**2 + 2 * f2**2  - f3**2 + 
                     K))) * tau3**2  + (a3**2 * f1**2  + a3 * f1**4 + 
               a3**2 * f2**2 + 4 * a3 * f1**2 * f2**2  + f1**4 * f2**2  + a3 * f2**4 + 
               f1**2 * f2**4 + 2 * a3 * f1**2 * f3**2  + f1**4 * f3**2 + 
               2 * a3 * f2**2 * f3**2 + 4 * f1**2 * f2**2 * f3**2  + f2**4 * f3**2 + 
               f1**2 * f3**4  + f2**2 * f3**4  + a2**2 * (f1**2  + f3**2) + 
               a1**2 * (f2**2  + f3**2)  + (f1**4  + f2**4 + 4 * f2**2 * f3**2  + f3**4 + 
                  4 * f1**2 * (f2**2  + f3**2) + 
                  a3 * (2 * (f1**2  + f2**2)  + f3**2)) * K  + (f1**2  + f2**2 + 
                  f3**2) * K**2 + 
               a1 * (2 * f1**2 * f2**2  + f2**4 + 2 * f1**2 * f3**2 + 4 * f2**2 * f3**2 + 
                  f3**4  + a3 * (f1**2 + 2 * f2**2  + f3**2) + 
                  a2 * (f1**2  + f2**2 + 2 * f3**2)  + (f1**2 + 
                     2 * (f2**2  + f3**2)) * K) + 
               a2 * (f1**4 + 2 * f2**2 * f3**2  + f3**4 + 
                  a3 * (2 * f1**2  + f2**2  + f3**2)  + f2**2 * K + 2 * f3**2 * K + 
                  2 * f1**2 * (f2**2 + 2 * f3**2  + K))) * tau3**4)))
   c6 =  tau2**4 * tau3**4 - 2 * tau1**2 * (-2 * tau2**2 * tau3**4  + tau2**4 * tau3**2 * (-2  + (3 * a1 + 2 * a2 + 2 * a3  - f1**2 + 
            2 * (f2**2  + f3**2  + K)) * tau3**2))  + tau1**4 * (tau3**4 + 
      2 * tau2**2 * tau3**2 * (2  - (2 * a1 + 3 * a2 + 2 * a3 + 2 * f1**2  - f2**2 + 
            2 * (f3**2  + K)) * tau3**2)  + tau2**4 * (1 - 
         2 * (2 * a1 + 2 * a2 + 3 * a3 + 2 * f1**2 + 2 * f2**2  - f3**2 + 
            2 * K) * tau3**2  + (a1**2  + a2**2  + a3**2 + 4 * a3 * f1**2  + f1**4 + 
            4 * a3 * f2**2 + 4 * f1**2 * f2**2  + f2**4 + 2 * a3 * f3**2 + 4 * f1**2 * f3**2 + 
            4 * f2**2 * f3**2  + f3**4 + 2 * (a3 + 2 * (f1**2  + f2**2  + f3**2)) * K + 
               K**2 + 2 * a2 * (a3 + 2 * f1**2  + f2**2 + 2 * f3**2  + K) + 
            2 * a1 * (a2  + a3  + f1**2 + 2 * (f2**2  + f3**2)  + K)) * tau3**4))
   c7 = 2 * tau1**2 * tau2**4 * tau3**4 + 2 * tau1**4 * tau2**2 * tau3**2 * (tau3**2  + tau2**2 * (1  - (a1 + 
               a2  + a3  + f1**2  + f2**2  + f3**2  + K) * tau3**2))
   c8 = tau1**4 * tau2**4 * tau3**4

   nroot = nroots(c0 + x*(c8*x**7 + c7*x**6 + c6*x**5 + c5*x**4 + c4*x**3 + c3*x**2 + c2*x + c1), n=13, maxsteps=1000)

   return np.complex64(nroot), np.array([k1, k2, k3, k4, k5]), np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8])

def msd_tri_oscexp_harm(a1,a2,a3,  tau1,tau2,tau3,  f1,f2,f3,  K, B, t):
   # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
   r_i, k_i, c_i = roots_tri_oscexp_harm(a1,a2,a3,  tau1,tau2,tau3,  f1,f2,f3,  K)

   summe = 0
   for i in range(len(r_i)):
      summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2 + k_i[3] * r_i[i]**3 + k_i[4] * r_i[i]**4)
   msd = B/c_i[-1] * summe
   return np.real(msd)


def vacf_tri_oscexp_harm(a1,a2,a3,  tau1,tau2,tau3,  f1,f2,f3,  K, B, t):
   # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
   r_i, k_i, c_i = roots_tri_oscexp_harm(a1,a2,a3,  tau1,tau2,tau3,  f1,f2,f3,  K)

   summe = 0
   for i in range(len(r_i)):
      summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2 + k_i[3] * r_i[i]**3 + k_i[4] * r_i[i]**4)
   vacf = B/c_i[-1] * summe / 2
   return np.real(vacf)


def pacf_tri_oscexp_harm(a1,a2,a3,  tau1,tau2,tau3,  f1,f2,f3,  K, B, t):
   # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
   r_i, k_i, c_i = roots_tri_oscexp_harm(a1,a2,a3,  tau1,tau2,tau3,  f1,f2,f3,  K)

   summe = 0
   for i in range(len(r_i)):
      summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2 + k_i[3] * r_i[i]**3 + k_i[4] * r_i[i]**4)
   pacf = -B/c_i[-1] * summe / 2
   return np.real(pacf)





def roots_osc_sincos_harm(a, b, f, tau, K):
    k1 = -2 * (1 + f**2 * tau**2) * (a + 2 * b * tau + a * f**2 * tau**2)
    k2 = 4 * a * tau**2 * (-1 + f**2 * tau**2)
    k3 = -2 * a * tau**4
    c0 = (K + f**2 * K * tau**2)**2
    c1 = 4 * b**2 * tau**2 + 4 * a * b * tau * (1 + f**2 * tau**2) + (a + a * f**2 * tau**2)**2 + 2 * K * (-1 + (3 * b - 2 * f**2 + K) * tau**2 - f**2 * (b + f**2 + K) * tau**4)
    c2 = 1 + 2 * (a**2 - 3 * b + f**2 - 2 * K) * tau**2 + (-2 * a**2 * f**2 + (b + f**2)**2 + 2 * (b + 2 * f**2) * K + K**2) * tau**4
    c3 = 2 + (a**2 - 2 * (b + f**2 + K)) * tau**2
    c4 = tau**4

    nroot = nroots(c0 + x*(c4*x**3 + c3*x**2 + c2*x + c1), n=15, maxsteps=1000)

    return np.complex64(nroot), np.array([k1, k2, k3]), np.array([c0, c1, c2, c3, c4])

def msd_osc_sincos_harm(a, b, f, tau,  K, B, t):
    # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
    r_i, k_i, c_i = roots_osc_sincos_harm(a, b, f, tau, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    msd = B/c_i[-1] * summe
    return np.real(msd)


def vacf_osc_sincos_harm(a, b, f, tau,  K, B, t):
    # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
    r_i, k_i, c_i = roots_osc_sincos_harm(a, b, f, tau, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    vacf = B/c_i[-1] * summe / 2
    return np.real(vacf)


def pacf_osc_sincos_harm(a, b, f, tau,  K, B, t):
    # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
    r_i, k_i, c_i = roots_osc_sincos_harm(a, b, f, tau, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    pacf = -B/c_i[-1] * summe / 2
    return np.real(pacf)


def msd_osc_cos(a, b, f, tau, B, t):
    # test whether same result is obtained when roots are numerically approximated instead of using analytic expression for roots
    a = np.array(a, dtype=np.float128)
    b = np.array(b, dtype=np.float128)#float(b)
    f = np.array(f, dtype=np.float128)#float(f)
    tau = np.array(tau, dtype=np.float128)#float(tau)
    B = np.array(B, dtype=np.float128)#float(B)
    k1 = -2* (1 + f**2 * tau**2) * (a + b*tau + a * f**2 * tau**2)
    k2 = -2*tau**2 * (b*tau + 2*a * (1 - f**2 * tau**2))
    k3 = -2 * a * tau**4
    c1 = (a + b*tau + a * f**2 * tau**2)**2
    c2 = 1 + 2*tau**2 * (a**2 - b + f**2) + 2*a*b*tau**3 + tau**4 * ((b + f**2)**2 - 2*a**2 * f**2)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2*(b + f**2)) )
    #print(c1,c2,c3,k1,k2,k3)
    
    nroot = nroots( tau**4 *x**3 + c3*x**2 + c2*x + c1, n=9, maxsteps=1000)
    nroot = np.complex128(nroot)

    # r1, r3, r5 = roots(c1, c2, c3, tau)
    # r1, r3, r5 = np.sqrt(nroot[0]), np.sqrt(nroot[2], np.sqrt(nroot[4]))
    r1sq, r3sq, r5sq = nroot[0], nroot[1], nroot[2]
    r_i = np.array([r1sq, r3sq, r5sq])
    c_i = np.array([c1, c2, c3])
    k_i = np.array([k1, k2, k3])

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])

    msd = B/tau**4 * (k_i[0] * t / np.prod(r_i) + summe)
    return np.real(msd)


def vacf_osc_analytic(a, b, f, tau, B, t):
    a = np.array(a, dtype=np.float128)
    b = np.array(b, dtype=np.float128)#float(b)
    f = np.array(f, dtype=np.float128)#float(f)
    tau = np.array(tau, dtype=np.float128)#float(tau)
    B = np.array(B, dtype=np.float128)#float(B)
    k1 = -2* (1 + f**2 * tau**2) * (a + b*tau + a * f**2 * tau**2)
    k2 = -2*tau**2 * (b*tau + 2*a * (1 - f**2 * tau**2))
    k3 = -2 * a * tau**4
    c1 = (a + b*tau + a * f**2 * tau**2)**2
    c2 = 1 + 2*tau**2 * (a**2 - b + f**2) + 2*a*b*tau**3 + tau**4 * ((b + f**2)**2 - 2*a**2 * f**2)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2*(b + f**2)) )
    #print(c1,c2,c3,k1,k2,k3)
    r1, r3, r5 = roots(c1, c2, c3, tau)
    #print(r1,r3,r5) #roots are same as in mathematica
    diff_r13 = r1**2 - r3**2
    diff_r15 = r1**2 - r5**2
    diff_r35 = r3**2 - r5**2
    

    I11 = -r1**2 * np.exp(-t*np.sqrt(-r1**2)) / (np.sqrt(-r1**2) * r1**2 * diff_r13 * diff_r15)
    I21 = I11 * r1**2 
    I31 = I21 * r1**2

    I12 = -r3**2 * np.exp(-np.sqrt(-r3**2)*t) / (-np.sqrt(-r3**2) * r3**2 * diff_r13 * diff_r35)
    I22 = I12 * r3**2 
    I32 = I22 * r3**2

    I13 = -r5**2 * np.exp(-np.sqrt(-r5**2)*t) / (np.sqrt(-r5**2) * r5**2 * diff_r15 * diff_r35)
    I23 = I13 * r5**2 
    I33 = I23 * r5**2

    I1 = k1 * (I11 + I12 + I13)
    I2 = k2 * (I21 + I22 + I23)
    I3 = k3* (I31 + I32 + I33)
    #print(B/tau**4, np.real(I1 + I2 + I3))
    vacf = B/tau**4 * np.real(I1 + I2 + I3)/2
    return vacf


def pacf_osc_cos(a, b, f, tau, B, t):
    # test whether same result is obtained when roots are numerically approximated instead of using analytic expression for roots
    a = np.array(a, dtype=np.float128)
    b = np.array(b, dtype=np.float128)#float(b)
    f = np.array(f, dtype=np.float128)#float(f)
    tau = np.array(tau, dtype=np.float128)#float(tau)
    B = np.array(B, dtype=np.float128)#float(B)
    k1 = -2* (1 + f**2 * tau**2) * (a + b*tau + a * f**2 * tau**2)
    k2 = -2*tau**2 * (b*tau + 2*a * (1 - f**2 * tau**2))
    k3 = -2 * a * tau**4
    c1 = (a + b*tau + a * f**2 * tau**2)**2
    c2 = 1 + 2*tau**2 * (a**2 - b + f**2) + 2*a*b*tau**3 + tau**4 * ((b + f**2)**2 - 2*a**2 * f**2)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2*(b + f**2)) )
    #print(c1,c2,c3,k1,k2,k3)
    
    nroot = nroots( tau**4 *x**3 + c3*x**2 + c2*x + c1, n=9, maxsteps=1000)
    nroot = np.complex128(nroot)

    # r1, r3, r5 = roots(c1, c2, c3, tau)
    # r1, r3, r5 = np.sqrt(nroot[0]), np.sqrt(nroot[2], np.sqrt(nroot[4]))
    r1sq, r3sq, r5sq = nroot[0], nroot[1], nroot[2]
    r_i = np.array([r1sq, r3sq, r5sq])
    c_i = np.array([c1, c2, c3])
    k_i = np.array([k1, k2, k3])

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])

    pacf = -B/tau**4 * (summe) / 2
    return np.real(pacf)



def roots_biexpo_delta(a, b, c, tau1, tau2):
    #roots of msd integral denominator for memory kernel of the form a*exp(-t/tau1) + b*exp(-t/tau2) + c*delta(t)
    k1 = -2 * c - 2 * a  * tau1 - 2 * b  * tau2
    k2 = -2 * (tau1  * tau2 * (b  * tau1 + a  * tau2) +  c * (tau1**2 + tau2**2))
    k3 = -2 * c  * tau1**2  * tau2**2 
    c1 = (c + a  * tau1 + b  * tau2)**2
    c2 = 1 + (-2 * a + c**2)  * tau1**2 + 2 * b * c  * tau1**2  * tau2 + (-2 * b + c**2 + 2 * a * c  * tau1 + (a + b)**2  * tau1**2)  * tau2**2
    c3 =  tau2**2 +  tau1**2 * (1 + (-2 * (a + b) + c**2)  * tau2**2)
    c4 =  tau1**2  * tau2**2
    
    nroot = nroots(c4*x**3 + c3*x**2 + c2*x + c1, n=15)

    return np.complex128(nroot), np.array([k1, k2, k3]), np.array([c1, c2, c3, c4])

def msd_biexpo_delta(a, b, c, tau1, tau2, B, t):
    r_i, k_i, c_i = roots_biexpo_delta(a, b, c, tau1, tau2)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])
    if k_i[0]/np.prod(r_i) < 0:
        print('diffusivity is smaller than zero')

    msd = B/c_i[-1] * (k_i[0] * t / np.prod(r_i) + summe)
    return np.real(msd)


def vacf_biexpo_delta(a, b, c, tau1, tau2, B, t):
    r_i, k_i, c_i = roots_biexpo_delta(a, b, c, tau1, tau2)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])
    if k_i[0]/np.prod(r_i) < 0:
        print('diffusivity is smaller than zero')

    vacf = B/c_i[-1] * summe / 2
    return np.real(vacf)


def pacf_biexpo_delta(a, b, c, tau1, tau2, B, t):
    r_i, k_i, c_i = roots_biexpo_delta(a, b, c, tau1, tau2)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])
    if k_i[0]/np.prod(r_i) < 0:
        print('diffusivity is smaller than zero')

    pacf = -B/c_i[-1] * (summe) / 2
    return np.real(pacf)


def roots_biexpo_delta_harm(a, b, c, tau1, tau2, K):
    #roots of msd integral denominator for memory kernel of the form a*exp(-t/tau1) + b*exp(-t/tau2) + c*delta(t) and harmonic potential with U(x)/m = K/2 * x**2
    k1 = -2 * (c + a * tau1 + b * tau2)
    k2 = -2 * (c * tau1**2 + b * tau1**2 * tau2 + c * tau2**2 +  a * tau1 * tau2**2)
    k3 = -2 * c * tau1**2 * tau2**2
    c0 = K**2
    c1 = c**2 - 2 * K + 2 * a * c * tau1 + a**2 * tau1**2 + 2 * a * K * tau1**2 + K**2 * tau1**2 + 2 * b * c * tau2 + 2 * a * b * tau1 * tau2 +  b**2 * tau2**2 + 2 * b * K * tau2**2 + K**2 * tau2**2
    c2 = 1 - 2 * a * tau1**2 + c**2 * tau1**2 - 2 * K * tau1**2 + 2 * b * c * tau1**2 * tau2 - 2 * b * tau2**2 + c**2 * tau2**2 - 2 * K * tau2**2 + 2 * a * c * tau1 * tau2**2 + a**2 * tau1**2 * tau2**2 \
        + 2 * a * b * tau1**2 * tau2**2 + b**2 * tau1**2 * tau2**2 + 2 * a * K * tau1**2 * tau2**2 + 2 * b * K * tau1**2 * tau2**2 + K**2 * tau1**2 * tau2**2
    c3 = tau2**2 + tau1**2 * (1 + (-2 * a - 2 * b + c**2 - 2 * K) * tau2**2)
    c4 = tau1**2 * tau2**2
    
    nroot = nroots(c0 + x*(c4*x**3 + c3*x**2 + c2*x + c1), n=15)

    return np.complex128(nroot), np.array([k1, k2, k3]), np.array([c0, c1, c2, c3, c4])

def msd_biexpo_delta_harm(a, b, c, tau1, tau2, K, B, t):
    r_i, k_i, c_i = roots_biexpo_delta_harm(a, b, c, tau1, tau2, K)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i]  + k_i[2] * r_i[i]**2)

    msd = B/c_i[-1] * summe
    return np.real(msd)

def vacf_biexpo_delta_harm(a, b, c, tau1, tau2, K, B, t):
    r_i, k_i, c_i = roots_biexpo_delta_harm(a, b, c, tau1, tau2, K)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i]  + k_i[2] * r_i[i]**2)

    vacf = B/c_i[-1] * summe / 2
    return np.real(vacf)

def aacf_biexpo_delta_harm(a, b, c, tau1, tau2, K, B, t,DT=None):
    r_i, k_i, c_i = roots_biexpo_delta_harm(a, b, c, tau1, tau2, K)
    summe = 0
    delta_arr = np.zeros(len(t))
    if DT is None:
        delta_arr[0]=1/(t[1]-t[0])
    else:
        delta_arr[0]=1/DT
    for i in range(len(r_i)):
        summe += ((np.exp(-np.sqrt(-r_i[i]) * t) * np.sqrt(-r_i[i])**3)-2*np.sqrt(-r_i[i])**2*delta_arr) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i]  + k_i[2] * r_i[i]**2)

    aacf = -B/c_i[-1] * summe / 2
    return np.real(aacf)

def pacf_biexpo_delta_harm(a, b, c, tau1, tau2, K, B, t):
    r_i, k_i, c_i = roots_biexpo_delta_harm(a, b, c, tau1, tau2, K)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i]  + k_i[2] * r_i[i]**2)

    msd = -B/c_i[-1] * summe / 2
    return np.real(msd)


def roots_tri_expo_harm(a, b, c, tau1, tau2, tau3, K):
    k1 = -2 * (a  * tau1 + b  * tau2 + c  * tau3)
    k2 = -2 * (b  * tau1**2  * tau2 + a  * tau1  * tau2**2 + c  * tau1**2  * tau3 + c  * tau2**2  * tau3 + a  * tau1  * tau3**2 + b  * tau2  * tau3**2)
    k3 = -2  * tau1  * tau2  * tau3 * (c  * tau1  * tau2 + b  * tau1  * tau3 + a  * tau2  * tau3)
    c0 = K**2
    c1 = (a * tau1 + b  * tau2 + c  * tau3)**2 + K**2 * (tau1**2 +  tau2**2 +  tau3**2) + 2 * K * (-1 + a  * tau1**2 + b  * tau2**2 + c  * tau3**2)
    c2 = 1 - 2 * K  * tau1**2 - 2 * b  * tau2**2 - 2 * K  * tau2**2 + b**2  * tau1**2  * tau2**2 + 2 * b * K  * tau1**2  * tau2**2 + K**2  * tau1**2  * tau2**2 + 2 * b * c  * tau1**2  * tau2  * tau3 \
        + (-2 * c - 2 * K + c**2  * tau1**2 + 2 * c * K  * tau1**2 + K**2  * tau1**2 + (b + c + K)**2  * tau2**2)  * tau3**2 + a**2  * tau1**2 * (tau2**2 +  tau3**2) \
        + 2 * a  * tau1 * (tau1 * (-1 + (b + K)  * tau2**2) + c  * tau2**2  * tau3 + ((c + K)  * tau1 + b  * tau2)  * tau3**2)
    c3 =  tau3**2 +  tau2**2 * (1 - 2 * (b + c + K)  * tau3**2) +  tau1**2 * (1 - 2 * (a + b + K)  * tau2**2 + (-2 * (a + c + K) + (a + b + c + K)**2  * tau2**2)  * tau3**2)
    c4 = tau2**2  * tau3**2 +  tau1**2 * (tau3**2 +  tau2**2 * (1 - 2 * (a + b + c + K) * tau3**2))
    c5 =  tau1**2 * tau2**2 * tau3**2

    nroot = nroots(c0 + x * (c5*x**4 + c4*x**3 + c3*x**2 + c2*x + c1), n=9, maxsteps=1000)

    return np.complex128(nroot), np.array([k1, k2, k3]), np.array([c0, c1, c2, c3, c4, c5])


def msd_tri_expo_harm(a1,a2,a3,  tau1,tau2,tau3, K, B, t):
    r_i, k_i, c_i = roots_tri_expo_harm(a1,a2,a3,  tau1,tau2,tau3, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    msd = B/c_i[-1] * summe
    return np.real(msd)


def vacf_tri_expo_harm(a1,a2,a3,  tau1,tau2,tau3, K, B, t):
    r_i, k_i, c_i = roots_tri_expo_harm(a1,a2,a3,  tau1,tau2,tau3, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    vacf = B/c_i[-1] * summe / 2
    return np.real(vacf)

def pacf_tri_expo_harm(a1,a2,a3,  tau1,tau2,tau3, K, B, t):
    r_i, k_i, c_i = roots_tri_expo_harm(a1,a2,a3,  tau1,tau2,tau3, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    pacf = -B/c_i[-1] * summe / 2
    return np.real(pacf)

def cxx_prime_tri_expo_harm(a1,a2,a3,  tau1,tau2,tau3, K, B, t):
    r_i, k_i, c_i = roots_tri_expo_harm(a1,a2,a3,  tau1,tau2,tau3, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (-np.exp(-np.sqrt(-r_i[i]) * t)) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    cxx_prime = -B/c_i[-1] * summe / 2
    return np.real(cxx_prime)


def roots_tri_expo(a, b, c, tau1, tau2, tau3):
    a = np.float128(a) 
    b = np.float128(b) 
    c = np.float128(c) 

    tau1 = np.float128(tau1) 
    tau2 = np.float128(tau2) 
    tau3 = np.float128(tau3)

    k1 = -2 * (a * tau1  + b * tau2  + c * tau3)
    k2 = -2 * (b * tau1**2 * tau2  + a * tau1 * tau2**2  + c * tau1**2 * tau3  + c * tau2**2 * tau3  + a * tau1 * tau3**2  + b * tau2 * tau3**2)
    k3 = -2 * tau1 * tau2 * tau3 * (c * tau1 * tau2  + b * tau1 * tau3  + a * tau2 * tau3)
    c1 =  (a * tau1  + b * tau2  + c * tau3)**2
    c2 = 1 - 2 * a * tau1**2 - 2 * b * tau2**2  + a**2 * tau1**2 * tau2**2 + 2 * a * b * tau1**2 * tau2**2  + b**2 * tau1**2 * tau2**2 + 2 * b * c * tau1**2 * tau2 * tau3 + 2 * a * c * tau1 * tau2**2 * tau3 - 2 * c * tau3**2  + a**2 * tau1**2 * tau3**2 + 2 * a * c * tau1**2 * tau3**2  + c**2 * tau1**2 * tau3**2 \
        + 2 * a * b * tau1 * tau2 * tau3**2  + b**2 * tau2**2 * tau3**2 + 2 * b * c * tau2**2 * tau3**2  + c**2 * tau2**2 * tau3**2
    c3 =  tau1**2  + tau2**2 - 2 * a * tau1**2 * tau2**2 - 2 * b * tau1**2 * tau2**2  + tau3**2 - 2 * a * tau1**2 * tau3**2 - 2 * c * tau1**2 * tau3**2 - 2 * b * tau2**2 * tau3**2 - 2 * c * tau2**2 * tau3**2  + a**2 * tau1**2 * tau2**2 * tau3**2 + 2 * a * b * tau1**2 * tau2**2 * tau3**2 \
        + b**2 * tau1**2 * tau2**2 * tau3**2 + 2 * a * c * tau1**2 * tau2**2 * tau3**2 + 2 * b * c * tau1**2 * tau2**2 * tau3**2  + c**2 * tau1**2 * tau2**2 * tau3**2
    c4 = -(-tau1**2 * tau2**2   -tau1**2 * tau3**2   -tau2**2 * tau3**2 + 2 * a * tau1**2 * tau2**2 * tau3**2 + 2 * b * tau1**2 * tau2**2 * tau3**2 + 2 * c * tau1**2 * tau2**2 * tau3**2)
    c5 =  tau1**2 * tau2**2 * tau3**2

    nroot = nroots(c5*x**4 + c4*x**3 + c3*x**2 + c2*x + c1, n=15)

    return np.complex128(nroot), np.array([k1, k2, k3]), np.array([c1, c2, c3, c4, c5])

def msd_tri_expo(a, b, c, tau1, tau2, tau3, B, t):
    r_i, k_i, c_i = roots_tri_expo(a, b, c, tau1, tau2, tau3)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])
    if -k_i[0]/np.prod(r_i) < 0:
        print('diffusivity is smaller than zero')
    msd = B/c_i[-1] * (-k_i[0] * t / np.prod(r_i) + summe)
    return np.real(msd)

def vacf_tri_expo(a, b, c, tau1, tau2, tau3, B, t):
    r_i, k_i, c_i = roots_tri_expo(a, b, c, tau1, tau2, tau3)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])

    vacf = B/c_i[-1] * summe / 2
    return np.real(vacf)

def pacf_tri_expo(a, b, c, tau1, tau2, tau3, B, t):
    r_i, k_i, c_i = roots_tri_expo(a, b, c, tau1, tau2, tau3)
    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])
    if -k_i[0]/np.prod(r_i) < 0:
        print('diffusivity is smaller than zero')
    pacf = -B/c_i[-1] *  summe / 2
    return np.real(pacf)


def cxx_prime_tri_expo(a, b, c, tau1, tau2, tau3, B, t):
    r_i, k_i, c_i = roots_tri_expo(a, b, c, tau1, tau2, tau3)
    summe = 0
    for i in range(len(r_i)):
        summe += (-np.exp(-np.sqrt(-r_i[i]) * t)) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])

    cxx_prime = -B/c_i[-1] * (summe) / 2
    return np.real(cxx_prime)



def roots_five_expo_harm(a, b, c, d, e,  tau1,tau2,tau3,tau4,tau5, K):
    a = np.float128(a) 
    b = np.float128(b) 
    c = np.float128(c) 
    d = np.float128(d)
    e = np.float128(e) 

    tau1 = np.float128(tau1) 
    tau2 = np.float128(tau2) 
    tau3 = np.float128(tau3)
    tau4 = np.float128(tau4) 
    tau5 = np.float128(tau5)
    K = np.float128(K)

    k1 = -2 * (a * tau1 + b * tau2 + c * tau3 + d * tau4 + e * tau5)

    k2 = -2 * (c * tau1**2 * tau3 + c * tau2**2 * tau3 + d * tau1**2 * tau4 + d * tau2**2 * tau4 + d * tau3**2 * tau4 + c * tau3 * tau4**2 + e * (tau1**2 + tau2**2 + tau3**2 + tau4**2) * tau5 + (c * tau3 + d * tau4) * tau5**2 + b * tau2 * (tau1**2 + tau3**2 + tau4**2 + tau5**2) + a * tau1 * (tau2**2 + tau3**2 + tau4**2 + tau5**2))

    k3 = -2 * (b * tau1**2 * tau2 * tau3**2 + a * tau1 * tau2**2 * tau3**2 + d * tau1**2 * tau2**2 * tau4 + d * tau1**2 * tau3**2 * tau4 + d * tau2**2 * tau3**2 * tau4 + b * tau1**2 * tau2 * tau4**2 + a * tau1 * tau2**2 * tau4**2 + a * tau1 * tau3**2 * tau4**2 + b * tau2 * tau3**2 * tau4**2 \
        + e * (tau3**2 * tau4**2 + tau2**2 * (tau3**2 + tau4**2) + tau1**2 * (tau2**2 + tau3**2 + tau4**2)) * tau5 + (d * (tau1**2 + tau2**2 + tau3**2) * tau4 + b * tau2 * (tau1**2 + tau3**2 + tau4**2) + a * tau1 * (tau2**2 + tau3**2 + tau4**2)) * tau5**2 + c * tau3 * (tau4**2 * tau5**2 + tau2**2 * (tau4**2 + tau5**2) + tau1**2 * (tau2**2 + tau4**2 + tau5**2)))

    k4 = -2 * tau1 * tau2 * tau3 * tau4 * (d * tau1 * tau2 * tau3 + (c * tau1 * tau2 + b * tau1 * tau3 + a * tau2 * tau3) * tau4) - 2 * e * (tau2**2 * tau3**2 * tau4**2 + tau1**2 * (tau3**2 * tau4**2 + tau2**2 * (tau3**2 + tau4**2))) * tau5 \
        -2 * (tau1 * tau2 * tau3 * (c * tau1 * tau2 + b * tau1 * tau3 + a * tau2 * tau3) + d * (tau2**2 * tau3**2 + tau1**2 * (tau2**2 + tau3**2)) * tau4 + (c * (tau1**2 + tau2**2) * tau3 + b * tau2 * (tau1**2 + tau3**2) + a * tau1 * (tau2**2 + tau3**2)) * tau4**2) * tau5**2

    k5 = -2 * tau1 * tau2 * tau3 * tau4 * tau5 * (e * tau1 * tau2 * tau3 * tau4 + d * tau1 * tau2 * tau3 * tau5 + c * tau1 * tau2 * tau4 * tau5 + b * tau1 * tau3 * tau4 * tau5 + a * tau2 * tau3 * tau4 * tau5)

    c0 = K**2
    c1 = (a * tau1 + b * tau2 + c * tau3 + d * tau4 + e * tau5)**2 + K**2 * (tau1**2 + tau2**2 + tau3**2 + tau4**2 + tau5**2) + 2 * K * (-1 + a * tau1**2 + b * tau2**2 + c * tau3**2 + d * tau4**2 + e * tau5**2)

    c2 = 1 - 2 * b * tau2**2 + b**2 * tau1**2 * tau2**2 + 2 * b * c * tau1**2 * tau2 * tau3 - 2 * c * tau3**2 + c**2 * tau1**2 * tau3**2 + b**2 * tau2**2 * tau3**2 + 2 * b * c * tau2**2 * tau3**2 + c**2 * tau2**2 * tau3**2 + 2 * b * d * tau1**2 * tau2 * tau4 + 2 * c * d * tau1**2 * tau3 * tau4 \
        + 2 * c * d * tau2**2 * tau3 * tau4 + 2 * b * d * tau2 * tau3**2 * tau4 - 2 * d * tau4**2 + d**2 * tau1**2 * tau4**2 + b**2 * tau2**2 * tau4**2 + 2 * b * d * tau2**2 * tau4**2 + d**2 * tau2**2 * tau4**2 + 2 * b * c * tau2 * tau3 * tau4**2 + c**2 * tau3**2 * tau4**2 \
        + 2 * c * d * tau3**2 * tau4**2 + d**2 * tau3**2 * tau4**2 + 2 * e * (d * (tau1**2 + tau2**2 + tau3**2) * tau4 + c * tau3 * (tau1**2 + tau2**2 + tau4**2) + b * tau2 * (tau1**2 + tau3**2 + tau4**2)) * tau5 + ((b * tau2 + c * tau3 + d * tau4)**2 \
        + e**2 * (tau1**2 + tau2**2 + tau3**2 + tau4**2) + 2 * e * (-1 + b * tau2**2 + c * tau3**2 + d * tau4**2)) * tau5**2 + a**2 * tau1**2 * (tau2**2 + tau3**2 + tau4**2 + tau5**2) + K**2 * (tau3**2 * tau4**2 + (tau3**2 + tau4**2) * tau5**2 \
        + tau2**2 * (tau3**2 + tau4**2 + tau5**2) + tau1**2 * (tau2**2 + tau3**2 + tau4**2 + tau5**2)) + 2 * K * (-tau4**2 + tau3**2 * (-1 + (c + d) * tau4**2) - tau5**2 + ((c + e) * tau3**2 + (d + e) * tau4**2) * tau5**2 + tau1**2 * (-1 + b * tau2**2 \
        + c * tau3**2 + d * tau4**2 + e * tau5**2) + tau2**2 * (-1 + b * tau3**2 + c * tau3**2 + b * tau4**2 + d * tau4**2 + (b + e) * tau5**2)) + 2 * a * tau1 * (b * tau2 * tau3**2 + d * tau2**2 * tau4 + d * tau3**2 * tau4 + b * tau2 * tau4**2 \
        + e * (tau2**2 + tau3**2 + tau4**2) * tau5 + (b * tau2 + d * tau4) * tau5**2 + c * tau3 * (tau2**2 + tau4**2 + tau5**2) + tau1 * (-1 + b * tau2**2 + c * tau3**2 + d * tau4**2 + e * tau5**2 + K * (tau2**2 + tau3**2 + tau4**2 + tau5**2)))

    c3 = tau4**2 + tau3**2 * (1 - 2 * (c + d + K) * tau4**2) + tau5**2 + (-2 * (c + e + K) * tau3**2 + (-2 * (d + e + K) + (c + d + e +  K)**2 * tau3**2) * tau4**2) * tau5**2 \
        + 2 * b * tau2 * tau3 * tau4 * tau5 * (e * tau3 * tau4 + d * tau3 * tau5 + c * tau4 * tau5) + 2 * a * tau1 * (tau2 * tau3 * tau4 * (d * tau2 * tau3 + c * tau2 * tau4 + b * tau3 * tau4) + e * (tau3**2 * tau4**2 + tau2**2 * (tau3**2 + tau4**2)) * tau5 + (d * (tau2**2 + tau3**2) * tau4 \
        + c * tau3 * (tau2**2 + tau4**2) + b * tau2 * (tau3**2 + tau4**2)) * tau5**2) + tau2**2 * (1 - 2 * K * tau3**2 - 2 * d * tau4**2 - 2 * K * tau4**2 + d**2 * tau3**2 * tau4**2 + 2 * d * K * tau3**2 * tau4**2 \
        + K**2 * tau3**2 * tau4**2 + 2 * d * e * tau3**2 * tau4 * tau5 + (-2 * e - 2 * K + e**2 * tau3**2 + 2 * e * K * tau3**2 + K**2 * tau3**2 + (d + e + K)**2 * tau4**2) * tau5**2 + c**2 * tau3**2 * (tau4**2 + tau5**2) + 2 * c * tau3 * (tau3 * (-1 + (d + K) * tau4**2) \
        + e * tau4**2 * tau5 + ((e + K) * tau3 + d * tau4) * tau5**2) + b**2 * (tau4**2 * tau5**2 + tau3**2 * (tau4**2 + tau5**2)) + 2 * b * (-tau4**2 + (-1 + (d + e + K) * tau4**2) * tau5**2 + tau3**2 * (-1 + (c + d + K) * tau4**2 + (c + e \
        + K) * tau5**2))) + tau1**2 * (1 - 2 * K * tau2**2 - 2 * c * tau3**2 - 2 * K * tau3**2 + c**2 * tau2**2 * tau3**2 + 2 * c * K * tau2**2 * tau3**2 + K**2 * tau2**2 * tau3**2 + 2 * c * d * tau2**2 * tau3 * tau4 - 2 * d * tau4**2 - 2 * K * tau4**2 \
        + d**2 * tau2**2 * tau4**2 + 2 * d * K * tau2**2 * tau4**2 + K**2 * tau2**2 * tau4**2 + c**2 * tau3**2 * tau4**2 + 2 * c * d * tau3**2 * tau4**2 + d**2 * tau3**2 * tau4**2 + 2 * c * K * tau3**2 * tau4**2 + 2 * d * K * tau3**2 * tau4**2 + K**2 * tau3**2 * tau4**2 \
        + 2 * e * (d * (tau2**2 + tau3**2) * tau4 + c * tau3 * (tau2**2 + tau4**2)) * tau5 + ((c * tau3 + d * tau4)**2 + e**2 * (tau2**2 + tau3**2 + tau4**2) + K**2 * (tau2**2 + tau3**2 + tau4**2) + 2 * K * (-1 + c * tau3**2 + d * tau4**2) \
        + 2 * e * (-1 + c * tau3**2 + d * tau4**2 + K * (tau2**2 + tau3**2 + tau4**2))) * tau5**2 + b**2 * tau2**2 * (tau3**2 + tau4**2 + tau5**2) + a**2 * (tau4**2 * tau5**2 + tau3**2 * (tau4**2 + tau5**2) + tau2**2 * (tau3**2 + tau4**2 + tau5**2)) \
        + 2 * b * tau2 * (tau3 * tau4 * (d * tau3 + c * tau4) + e * (tau3**2 + tau4**2) * tau5 + (c * tau3 + d * tau4) * tau5**2 + tau2 * (-1 + c * tau3**2 + K * tau3**2 + d * tau4**2 + K * tau4**2 + (e + K) * tau5**2)) + 2 * a * (-tau4**2 + (-1 + (d + e \
        + K) * tau4**2) * tau5**2 + tau2**2 * (-1 + b * tau3**2 + c * tau3**2 + K * tau3**2 + b * tau4**2 + d * tau4**2 + K * tau4**2 + (b + e + K) * tau5**2) + tau3**2 * (-1 + (c +d + K) * tau4**2 + (c + e + K) * tau5**2)))

    c4 = tau3**2 * tau4**2 + (tau4**2 + tau3**2 * (1 - 2 * (c + d + e + K) * tau4**2)) * tau5**2 + 2 * a * tau1 * tau2 * tau3 * tau4 * tau5 * (e * tau2 * tau3 * tau4 + (d * tau2 * tau3 + c * tau2 * tau4 + b * tau3 * tau4) * tau5) \
        + tau2**2 * (tau4**2 + (1 - 2 * (b + d + e + K) * tau4**2) * tau5**2 + tau3**2 * (1 - 2 * (b + c + d + K) * tau4**2 + (-2 * (b + c + e + K) + (b + c + d + e + K)**2 * tau4**2) * tau5**2)) \
        + tau1**2 * (tau3**2 + tau4**2 - 2 * (a + c + d + K) * tau3**2 * tau4**2 + (1 - 2 * (a + c + e + K) * tau3**2 + (-2 * (a + d + e + K) + (a + c + d + e + K)**2 * tau3**2) * tau4**2) * tau5**2 \
        + 2 * b * tau2 * tau3 * tau4 * tau5 * (e * tau3 * tau4 + d * tau3 * tau5 + c * tau4 * tau5) + tau2**2 * (1 - 2 * a * tau3**2 - 2 * b * tau3**2 - 2 * c * tau3**2 - 2 * K * tau3**2 - 2 * a * tau4**2 - 2 * b * tau4**2 \
        - 2 * d * tau4**2 - 2 * K * tau4**2 + a**2 * tau3**2 * tau4**2 + 2 * a * b * tau3**2 * tau4**2 + b**2 * tau3**2 * tau4**2 + 2 * a * c * tau3**2 * tau4**2 + 2 * b * c * tau3**2 * tau4**2 + c**2 * tau3**2 * tau4**2 + 2 * a * d * tau3**2 * tau4**2 \
        + 2 * b * d * tau3**2 * tau4**2 + 2 * c * d * tau3**2 * tau4**2 + d**2 * tau3**2 * tau4**2 + 2 * a * K * tau3**2 * tau4**2 + 2 * b * K * tau3**2 * tau4**2 + 2 * c * K * tau3**2 * tau4**2 + 2 * d * K * tau3**2 * tau4**2 + K**2 * tau3**2 * tau4**2 + 2 * e * tau3 * tau4 * (d * tau3 \
        + c * tau4) * tau5 + (-2 * (b + e + K) + (b + c + e + K)**2 * tau3**2 + 2 * c * d * tau3 * tau4 + (b + d + e + K)**2 * tau4**2 + a**2 * (tau3**2 + tau4**2) + 2 * a * (-1 + (b + c + e + K) * tau3**2 + (b + d + e + K) * tau4**2)) * tau5**2))

    c5 = tau3**2 * tau4**2 * tau5**2 + tau2**2 * (tau3**2 * tau4**2 + (tau3**2 + (1 - 2 * (b + c + d + e + K) * tau3**2) * tau4**2) * tau5**2) + tau1**2 * (tau3**2 * tau4**2 + (tau4**2 + tau3**2 * (1 - 2 * (a + c + d + e + K) * tau4**2)) * tau5**2 \
        + tau2**2 * (tau4**2 + (1 - 2 * (a + b + d + e + K) * tau4**2) * tau5**2 + tau3**2 * (1 - 2 * (a + b + c + d + K) * tau4**2 + (-2 * (a + b + c + e + K) + (a + b + c + d + e + K)**2 * tau4**2) * tau5**2)))

    c6 = tau2**2 * tau3**2 * tau4**2 * tau5**2 + tau1**2 * (tau3**2 * tau4**2 * tau5**2 + tau2**2 * (tau3**2 * tau4**2 + (tau3**2 + (1 - 2 * (a + b + c + d + e + K) * tau3**2) * tau4**2) * tau5**2))

    c7 = tau1**2 * tau2**2 * tau3**2 * tau4**2 * tau5**2

    try:
        nroot = nroots(c0 + x * (c7 *x**6 + c6*x**5 + c5*x**4 + c4*x**3 + c3*x**2 + c2*x + c1), n=9, maxsteps=1000)
    except:
        nroot = nroots(c0 + x * (c7 *x**6 + c6*x**5 + c5*x**4 + c4*x**3 + c3*x**2 + c2*x + c1), n=5, maxsteps=1000)

    return np.complex128(nroot),  np.array([k1, k2, k3, k4, k5]), np.array([c0, c1, c2, c3, c4, c5, c6, c7])

def msd_five_expo_harm(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, K, B, t):
    r_i, k_i, c_i = roots_five_expo_harm(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2 + k_i[3] * r_i[i]**3 + k_i[4] * r_i[i]**4)
    msd = B/c_i[-1] * summe
    return np.real(msd)

def vacf_five_expo_harm(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, K, B, t):
    r_i, k_i, c_i = roots_five_expo_harm(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2 + k_i[3] * r_i[i]**3 + k_i[4] * r_i[i]**4)
    vacf = B/c_i[-1] * summe / 2
    return np.real(vacf)

def pacf_five_expo_harm(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, K, B, t):
    r_i, k_i, c_i = roots_five_expo_harm(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2 + k_i[3] * r_i[i]**3 + k_i[4] * r_i[i]**4)
    pacf = -B/c_i[-1] * summe / 2
    return np.real(pacf)

def cxx_prime_five_expo_harm(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, K, B, t):
    r_i, k_i, c_i = roots_five_expo_harm(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (-np.exp(-np.sqrt(-r_i[i]) * t)) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2 + k_i[3] * r_i[i]**3 + k_i[4] * r_i[i]**4)
    cxx_prime = -B/c_i[-1] * summe / 2
    return np.real(cxx_prime)




def roots_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5):
   a1 = np.float128(a1) #np.array(a1, dtype=np.float128)
   a2 = np.float128(a2) #np.array(a2, dtype=np.float128)
   a3 = np.float128(a3) #np.array(a3, dtype=np.float128)
   a4 = np.float128(a4) #np.array(a4, dtype=np.float128)
   a5 = np.float128(a5) #np.array(a5, dtype=np.float128)

   tau1 = np.float128(tau1) #np.array(tau1, dtype=np.float128)
   tau2 = np.float128(tau2) #np.array(tau2, dtype=np.float128)
   tau3 = np.float128(tau3) #np.array(tau3, dtype=np.float128)
   tau4 = np.float128(tau4) #np.array(tau4, dtype=np.float128)
   tau5 = np.float128(tau5) #np.array(tau5, dtype=np.float128)

   # a1 = np.array(a1, dtype=np.float64)
   # a2 = np.array(a2, dtype=np.float64)
   # a3 = np.array(a3, dtype=np.float64)
   # a4 = np.array(a4, dtype=np.float64)
   # a5 = np.array(a5, dtype=np.float64)

   # tau1 = np.array(tau1, dtype=np.float64)
   # tau2 = np.array(tau2, dtype=np.float64)
   # tau3 = np.array(tau3, dtype=np.float64)
   # tau4 = np.array(tau4, dtype=np.float64)
   # tau5 = np.array(tau5, dtype=np.float64)


   k_1 = -2 * (a1 * tau1 + a2 * tau2 + a3 * tau3 + a4 * tau4 + a5 * tau5)

   k_2 = -2 * (a3 * tau1**2 * tau3 + a3 * tau2**2 * tau3 + a4 * tau1**2 * tau4 + a4 * tau2**2 * tau4 + a4 * tau3**2 * tau4 + a3 * tau3 * tau4**2 + a5 * (tau1**2 + tau2**2 + tau3**2 + tau4**2) * tau5 + (a3 * tau3 + a4 * tau4) * tau5**2 + a2 * tau2 * (tau1**2 + tau3**2 + tau4**2 + tau5**2) + a1 * tau1 * (tau2**2 + tau3**2 + tau4**2 + tau5**2))
   
   k_3 = -2 * (a2 * tau1**2 * tau2 * tau3**2 + a1 * tau1 * tau2**2 * tau3**2 + a4 * tau1**2 * tau2**2 * tau4 + a4 * tau1**2 * tau3**2 * tau4 + a4 * tau2**2 * tau3**2 * tau4 + a2 * tau1**2 * tau2 * tau4**2 + a1 * tau1 * tau2**2 * tau4**2 + a1 * tau1 * tau3**2 * tau4**2 + a2 * tau2 * tau3**2 * tau4**2 + a5 * (tau3**2 * tau4**2 + tau2**2 * (tau3**2 + tau4**2) + \
      tau1**2 * (tau2**2 + tau3**2 + tau4**2)) * tau5 + (a4 * (tau1**2 + tau2**2 + tau3**2) * tau4 + a2 * tau2 * (tau1**2 + tau3**2 + tau4**2) + a1 * tau1 * (tau2**2 + tau3**2 + tau4**2)) * tau5**2 +a3 * tau3 * (tau4**2 * tau5**2 + tau2**2 * (tau4**2 + \
      tau5**2) + tau1**2 * (tau2**2 + tau4**2 + tau5**2)))
   k_4 = -2 * (tau1 * tau2 * tau3 * tau4 * (a4 * tau1 * tau2 * tau3 + (a3 * tau1 * tau2 + a2 * tau1 * tau3 + a1 * tau2 * tau3) * tau4) + a5 * (tau2**2 * tau3**2 * tau4**2 + tau1**2 * (tau3**2 * tau4**2 + tau2**2 * (tau3**2 + tau4**2))) * tau5 + (tau1 * tau2 * tau3 * (a3 * tau1 * tau2 + a2 * tau1 * tau3 + a1 * tau2 * tau3) \
      + a4 * (tau2**2 * tau3**2 + tau1**2 * (tau2**2 + tau3**2)) * tau4 + (a3 * (tau1**2 + tau2**2) * tau3 + a2 * tau2 * (tau1**2 + tau3**2) + a1 * tau1 * (tau2**2 + tau3**2)) * tau4**2) * tau5**2)
   k_5 = -2 * tau1 * tau2 * tau3 * tau4 * tau5 * (a5 * tau1 *tau2 * tau3 * tau4 + a4 * tau1 * tau2 * tau3 * tau5 + (a3 * tau1 * tau2 + a2 * tau1 * tau3 + a1 * tau2 * tau3) * tau4 * tau5)
       
   c_1 =  (a1 * tau1 + a2 * tau2 + a3 * tau3 + a4 * tau4 + a5 * tau5)**2

   c_2 = 1 - 2 * a2 * tau2**2 - 2 * a3 * tau3**2 + a3**2 * tau1**2 * tau3**2 + a3**2 * tau2**2 * tau3**2 + 2 * a3 * a4 * tau1**2 * tau3 * tau4 + 2 * a3 * a4 * tau2**2 * tau3 * tau4 - 2 * a4 * tau4**2 + a4**2 * tau1**2 * tau4**2 + a4**2 * tau2**2 * tau4**2 + a3**2 * tau3**2 * tau4**2 + 2 * a3 * a4 * tau3**2 * tau4**2 + a4**2 * tau3**2 * tau4**2 + 2 * a5 * (a4 * (tau1**2 + tau2**2 + tau3**2) * tau4 + \
      a3 * tau3 * (tau1**2 + tau2**2 + tau4**2)) * tau5 + ((a3 * tau3 + a4 * tau4)**2 + a5**2 * (tau1**2 + tau2**2 + tau3**2 + tau4**2) + 2 * a5 * (-1 + a3 * tau3**2 + a4 * tau4**2)) * tau5**2 + a2**2 * tau2**2 * (tau1**2 + tau3**2 + tau4**2 + tau5**2) + a1**2 * tau1**2 * (tau2**2 + tau3**2 + tau4**2 + tau5**2) + 2 * a2 * tau2 * (a5 * tau5 * (tau1**2 + tau3**2 + tau4**2 \
      +  tau2 * tau5) + a4 * tau4 * (tau1**2 + tau3**2 + tau2 * tau4 + tau5**2) + a3 * tau3 * (tau1**2 + tau2 * tau3 + tau4**2 + tau5**2)) + 2 * a1 * tau1 * (a2 * tau2 * tau3**2 + a4 * tau2**2 * tau4 + a4 * tau3**2 * tau4 + a2 * tau2 * tau4**2 + a5 * (tau2**2 + tau3**2 + tau4**2) * tau5 + (a2 * tau2 + a4 * tau4) * tau5**2 + a3 * tau3 * (tau2**2 + tau4**2 + tau5**2) + tau1 * (-1 + a2 * tau2**2 + a3 * tau3**2 + a4 * tau4**2 + a5 * tau5**2))

   c_3 = tau4**2 + tau3**2 * (1 - 2 * (a3 + a4) * tau4**2) + tau5**2 + (-2 * (a3 + a5) * tau3**2 + (-2 * (a4 + a5) + (a3 + a4 + a5)**2 * tau3**2) * tau4**2) * tau5**2 + 2 * a2 * tau2 * tau3 * tau4 * tau5 * (a5 * tau3 * tau4 + a4 * tau3 * tau5 + a3 * tau4 * tau5) + 2 * a1 * tau1 * (tau2 * tau3 * tau4 * (a4 * tau2 * tau3 \
      + a3 * tau2 * tau4 + a2 * tau3 * tau4) + a5 * (tau3**2 * tau4**2 + tau2**2 * (tau3**2 + tau4**2)) * tau5 + (a4 * (tau2**2 + tau3**2) * tau4 + a3 * tau3 * (tau2**2 + tau4**2) + a2 * tau2 * (tau3**2 + tau4**2)) * tau5**2) + tau2**2 * (1 - 2 * a4 * tau4**2 + tau3**2 * (-2 * a3 + (a3 + a4)**2 * tau4**2) + 2 * a5 * tau3 * tau4 * (a4 * tau3 \
      + a3 * tau4) * tau5 + ((a3 * tau3 + a4 * tau4)**2 + a5**2 * (tau3**2 + tau4**2) + 2 * a5 * (-1 + a3 * tau3**2 + a4 * tau4**2)) * tau5**2 + a2**2 * (tau4**2 * tau5**2 + tau3**2 * (tau4**2 + tau5**2)) + 2 * a2 * (-tau4**2 + (-1 + (a4 + a5) * tau4**2) * tau5**2 + tau3**2 * (-1 + (a3 + a4) * tau4**2 + (a3 + a5) * tau5**2))) + tau1**2 * (1 - 2 * a3 * tau3**2 \
      + a3**2 * tau2**2 * tau3**2 + 2 * a3 * a4 * tau2**2 * tau3 * tau4 - 2 * a4 * tau4**2 + a4**2 * tau2**2 * tau4**2 + a3**2 * tau3**2 * tau4**2 + 2 * a3 * a4 * tau3**2 * tau4**2 + a4**2 * tau3**2 * tau4**2 + 2 * a5 * (a4 * (tau2**2 + tau3**2) * tau4 + a3 * tau3 * (tau2**2 + tau4**2)) * tau5 + ((a3 * tau3 + a4 * tau4)**2 + a5**2 * (tau2**2 + tau3**2 + tau4**2) \
      + 2 * a5 * (-1 + a3 * tau3**2 + a4 * tau4**2)) * tau5**2 + a2**2 * tau2**2 * (tau3**2 + tau4**2 + tau5**2) + a1**2 * (tau4**2 * tau5**2 + tau3**2 * (tau4**2 + tau5**2) + tau2**2 * (tau3**2 + tau4**2 + tau5**2)) + 2 * a2 * tau2 * (tau3 * tau4 * (a4 * tau3 + a3 * tau4) +a5 * (tau3**2 + tau4**2) * tau5 + (a3 * tau3 + a4 * tau4) * tau5**2 + tau2 * (-1 + \
      a3 * tau3**2 + a4 * tau4**2 + a5 * tau5**2)) + 2 * a1 * (-tau4**2 + (-1 + (a4 + a5) * tau4**2) * tau5**2 + tau3**2 * (-1 + (a3 + a4) * tau4**2 + (a3 + a5) * tau5**2) + tau2**2 * (-1 + a3 * tau3**2 + a4 * tau4**2 + a5 * tau5**2 + a2 * (tau3**2 + tau4**2 + tau5**2))))
           
   c_4 = tau3**2 * tau4**2 + (tau4**2 + tau3**2 * (1 - 2 * (a3 + a4 + a5) * tau4**2)) * tau5**2 + 2 * a1 * tau1 * tau2 * tau3 * tau4 * tau5 * (a5 * tau2 * tau3 * tau4 + (a4 * tau2 * tau3 + a3 * tau2 * tau4 + a2 * tau3 * tau4) * tau5) + tau2**2 * (tau5**2 + tau4**2 * (1 - 2 * (a2 + a4 + a5) * tau5**2) + tau3**2 * (1 - 2 * (a2 + a3 + \
         a4) * tau4**2 + (-2 * (a2 + a3 + a5) + (a2 + a3 + a4 + a5)**2 * tau4**2) * tau5**2)) + tau1**2 * (tau3**2 + tau4**2 - 2 * (a1 + a3 + a4) * tau3**2 * tau4**2 + (1 - 2 * (a1 + a3 + a5) * tau3**2 + (-2 * (a1 + a4 + a5) + (a1 + a3 + a4 + a5)**2 * tau3**2) * tau4**2) * tau5**2 + 2 * a2 * tau2 * tau3 * tau4 * tau5 * (a5 * tau3 * tau4 + a4 * tau3 * tau5 + a3 * tau4 * tau5) + tau2**2 * (1 - \
         2 * a4 * tau4**2 + tau3**2 * (-2 * a3 + (a3 + a4)**2 * tau4**2) + 2 * a5 * tau3 * tau4 * (a4 * tau3 + a3 * tau4) * tau5 + ((a3 * tau3 + a4 * tau4)**2 + a5**2 * (tau3**2 + tau4**2) + 2 * a5 * (-1 + a3 * tau3**2 + a4 * tau4**2)) * tau5**2 + a1**2 * (tau4**2 * tau5**2 + tau3**2 * (tau4**2 + tau5**2)) + a2**2 * (tau4**2 * tau5**2 + tau3**2 * (tau4**2 + tau5**2)) + \
         2 * a2 * (-tau4**2 + (-1 + (a4 + a5) * tau4**2) * tau5**2 + tau3**2 * (-1 + (a3 + a4) * tau4**2 + (a3 + a5) * tau5**2)) + 2 * a1 * (-tau4**2 + (-1 + (a2 + a4 + a5) * tau4**2) * tau5**2 + tau3**2 * (-1 + (a2 + a3 + a4) * tau4**2 + (a2 + a3 + a5) * tau5**2))))
            
   c_5 = tau3**2 * tau4**2 * tau5**2 + tau2**2 * (tau3**2 * tau4**2 + (tau3**2 + (1 - 2 * (a2 + a3 + a4 + a5) * tau3**2) * tau4**2) * tau5**2) + tau1**2 * (tau3**2 * tau4**2 + (tau4**2 + tau3**2 * (1 - 2 * (a1 + a3 + a4 + a5) * tau4**2)) * tau5**2 + tau2**2 * (tau4**2 + (1 - 2 * (a1 + a2 + a4 + a5) * tau4**2) * tau5**2 + tau3**2 * (1 - \
      2 * (a1 + a2 + a3 + a4) * tau4**2 + (-2 * (a1 + a2 + a3 + a5) + (a1 + a2 + a3 + a4 + a5)**2 * tau4**2) * tau5**2)))
             
   c_6 = tau2**2 * tau3**2 * tau4**2 * tau5**2 + tau1**2 * (tau3**2 * tau4**2 * tau5**2 + tau2**2 * (tau3**2 * tau4**2 + (tau3**2 + (1 - 2 * (a1 + a2 + a3 + a4 + a5) * tau3**2) * tau4**2) * tau5**2))

   c_7 = tau1**2 * tau2**2 * tau3**2 * tau4**2 * tau5**2

   #print(k_1, k_2, k_3, k_4, k_5, c_1, c_2, c_3, c_4, c_5, c_6, c_7)

   try:
      nroot = nroots(c_7 *x**6 + c_6*x**5 + c_5*x**4 + c_4*x**3 + c_3*x**2 + c_2*x + c_1, n=9, maxsteps=1000)
   except:
      try:
         nroot = nroots(c_7 *x**6 + c_6*x**5 + c_5*x**4 + c_4*x**3 + c_3*x**2 + c_2*x + c_1, n=7, maxsteps=1000)
      except:
         nroot = nroots(c_7 *x**6 + c_6*x**5 + c_5*x**4 + c_4*x**3 + c_3*x**2 + c_2*x + c_1, n=5, maxsteps=1000)
   return np.complex128(nroot), np.array([k_1, k_2, k_3, k_4, k_5]), np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7])


def msd_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, B, t):
   r_i, k_i, c_i = roots_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5)

   summe = 0 #np.zeros(len(t))
   for i in range(len(r_i)):
      summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i] + k_i[3] * r_i[i]**2 + k_i[4] * r_i[i]**3)
   
   if -k_i[0]/np.prod(r_i) < 0:
      print('diffusivity is smaller than zero')
   msd = B/c_i[-1] * (-k_i[0] * t / np.prod(r_i) + summe)

   return np.real(msd)


def vacf_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, B, t):
   r_i, k_i, c_i = roots_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5)

   summe = 0 #np.zeros(len(t))
   for i in range(len(r_i)):
      summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / ( np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i] + k_i[3] * r_i[i]**2 + k_i[4] * r_i[i]**3)
   
   vacf = B/c_i[-1] *  summe / 2

   return np.real(vacf)



def pacf_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, B, t):
   r_i, k_i, c_i = roots_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5)

   summe = 0 #np.zeros(len(t))
   for i in range(len(r_i)):
      summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i] + k_i[3] * r_i[i]**2 + k_i[4] * r_i[i]**3)
   
   if -k_i[0]/np.prod(r_i) < 0:
      print('diffusivity is smaller than zero')
   pacf = -B/c_i[-1] * (summe)  /2

   return np.real(pacf)

def cxx_prime_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5, B, t):
   r_i, k_i, c_i = roots_five_expo(a1,a2,a3,a4,a5,  tau1,tau2,tau3,tau4,tau5)

   summe = 0 #np.zeros(len(t))
   for i in range(len(r_i)):
      summe += (-np.exp(-np.sqrt(-r_i[i]) * t)) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i] + k_i[3] * r_i[i]**2 + k_i[4] * r_i[i]**3)
   
   if -k_i[0]/np.prod(r_i) < 0:
      print('diffusivity is smaller than zero')
   cxx_prime = -B/c_i[-1] * (summe)  /2

   return np.real(cxx_prime)




def cost_msd_expo(args, msd_data, t, m=31.1, num_expo=5, msd_fun=msd_five_expo):
    gamma = args[:num_expo]
    tau = args[num_expo : 2*num_expo]
    amps = gamma / m / tau
    resid_vec = np.log10(msd_fun(*amps, *args[num_expo:], t)) - np.log10(msd_data) # use logarithm, such that not only long time behavior is fitted accurately
    if np.any(np.isinf(resid_vec)) or np.any(np.isnan(resid_vec)):
        resid_vec = np.ones(len(resid_vec)) * 1e140
    return resid_vec

def opt_msd_expo(msd_data, t, m, lb=[], upb=[], x0=[], msd_fun=msd_five_expo):
    if msd_fun == msd_five_expo:
        num_expo = 5
        if len(lb) == 0:
            lb = np.array([1, 1, 1, 1, 1,  1e-4, 1e-4, 1e-4, 1e-4, 1e-4,  1e-6])
        if len(upb) == 0:
            upb = np.array([1e13, 1e13, 1e13, 1e13, 1e13,  1e5, 1e5, 1e5, 1e5, 1e5,  2e5])

    elif msd_fun == msd_five_expo_harm:
        num_expo = 5
        if len(lb) == 0:
            lb = np.array([1, 1, 1, 1, 1,  1e-4, 1e-4, 1e-4, 1e-4, 1e-4,  1e-2,  1e-6])
        if len(upb) == 0:
            upb = np.array([1e13, 1e13, 1e13, 1e13, 1e13,  1e5, 1e5, 1e5, 1e5, 1e5,  100,  2e5])

    elif msd_fun == msd_tri_expo:
        num_expo = 3
        if len(lb) == 0:
            lb = np.array([1, 1, 1,  1e-4, 1e-4, 1e-4,  1e-6])
        if len(upb) == 0:
            upb = np.array([1e13, 1e13, 1e13,  1e5, 1e5, 1e5,  2e5])

    elif msd_fun == msd_tri_expo_harm:
        num_expo = 3
        if len(lb) == 0:
            lb = np.array([1, 1, 1,  1e-4, 1e-4, 1e-4,  1e-2,  1e-6])
        if len(upb) == 0:
            upb = np.array([1e13, 1e13, 1e13,  1e5, 1e5, 1e5,  100,  2e5])

    if len(x0) == 0:
        x0 = np.random.rand(len(upb)) * upb
    
    res=least_squares(lambda x: cost_msd_expo(x, msd_data, t, m, num_expo=num_expo, msd_fun=msd_fun), x0, bounds=(lb,upb))
    
    args = res.x
    gamma = args[:num_expo]
    tau = args[num_expo : 2*num_expo]
    amps = gamma / m / tau
    result = np.array([*amps, *args[num_expo:]])

    return result, res.cost