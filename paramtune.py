import jax.numpy as jnp
from timing import timing_van_jax
import scipy.optimize as opt
import matplotlib.pyplot as plt
from jax import jacfwd, jacrev
from polyfit import Polyfit
import json

class Paramtune:
    def __init__(self, target_json, initial_guess, pcoeffs_npz, chi2res_npz, **kwargs):
        with open(target_json, 'r') as f:
            target_data = json.loads(f.read())
        target_values = {k: target_data[k][0] for k in target_data}
        target_error = {k: target_data[k][1] for k in target_data}

        mc_fits = Polyfit(pcoeffs_npz, chi2res_npz, **kwargs)
        args = (list(target_values.values()), list(target_error.values()),
                list(mc_fits.pcoeffs.values()), list(mc_fits.cov.values()))
        print(jnp.array(list(mc_fits.pcoeffs.values())).shape)
        p_opt = opt.minimize(self.objective_func, initial_guess, args = args, method='Nelder-Mead')
        print(p_opt)

    def objective_func(self, p, d, d_sig, coeff, cov):
        sum_over = 0
        poly = timing_van_jax([p], 3)[0]

        #Loop over the bins
        for i in range(jnp.size(jnp.array(coeff), axis=0)):
            #We don't want to divide by 0. For real data, the error might be zero because no events were observered in the bin.
            if d_sig[i] == 0.0:
                continue
            f_sig = jnp.sqrt(jnp.matmul(poly, jnp.matmul(cov[i], poly.T))) #Finding uncertainty of surrogate function at point p
            adj_res_sq = (d[i]-jnp.matmul(coeff[i], poly.T))**2/(d_sig[i]**2 + f_sig**2) #Inner part of summation
            sum_over = sum_over + adj_res_sq
        return sum_over

    def objective_func_no_err(self, p, d, d_sig, coeff):
        sum_over = 0
        poly = timing_van_jax([p], 3)[0]

        #Loop over the bins
        for i in range(jnp.size(jnp.array(coeff), axis=0)):
            if d_sig[i] == 0.0:
                continue
            adj_res_sq = (d[i]-jnp.matmul(coeff[i], poly.T))**2/(d_sig[i]**2) #Inner part of summation
            sum_over = sum_over + adj_res_sq
        return sum_over