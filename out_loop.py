import jax.numpy as jnp
import h5py
from timing import timing_van_jax

def objective_func(p, d, d_sig, coeff, cov):
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

def objective_func_no_err(p, d, d_sig, coeff):
    sum_over = 0
    poly = timing_van_jax([p], 3)[0]

    #Loop over the bins
    for i in range(jnp.size(jnp.array(coeff), axis=0)):
        if d_sig[i] == 0.0:
            continue
        adj_res_sq = (d[i]-jnp.matmul(coeff[i], poly.T))**2/(d_sig[i]**2) #Inner part of summation
        sum_over = sum_over + adj_res_sq
    return sum_over