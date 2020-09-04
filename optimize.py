import numpy as np
from scipy.optimize import differential_evolution as de
from scipy.optimize import NonlinearConstraint as nlc

def kf(x, w, freq):
    kc = x>0
    kw = ~np.any(w[~kc,:], axis=0)
    return -freq[kw].sum()

def cons_fun(x):
    return np.sum(x>0)

def optimize(w, freq):
    cons = nlc(cons_fun, -np.inf, 1000)
    bnds = [np.array([-1,1]),]*w.shape[0]
    res = de(kf, args=(w, freq), maxiter=1000, bounds=bnds, popsize=2, polish=False, constraints=cons, disp=True, workers=-1, updating='deferred')
    output = res.x>0
    np.save('output.npy', output)

if __name__ == '__main__':
    # try optimizing toy version of problem
    small_w = np.load('small_w.npy')
    small_freq = np.load('small_freq.npy')
    optimize(small_w, small_freq)
    
    # try optimizing actual problem
    w = np.load('w.npy')
    freq = np.load('freq.npy')
    optimize(w, freq)
