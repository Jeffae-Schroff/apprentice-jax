import jax.numpy as jnp
import h5py
from timing import poly_gen_D, timing_van_jax

dummy_cov_arr = jnp.load("dummy_cov.npz")
dummy_chi2res = jnp.load("dummy_chi2res.npz")
dummy_pcoeffs = jnp.load("dummy_pceoffs.npz")
in_data = h5py.File('inputdata.h5', "r+")

cov = dict(zip((range(jnp.size(dummy_cov_arr))), (dummy_cov_arr[k] for k in dummy_cov_arr)))
coeff = dict(zip((range(jnp.size(dummy_pcoeffs))), (dummy_pcoeffs[k] for k in dummy_pcoeffs)))


#Gives the objective function given target frequencies d, uncertainty d_sig, and tuning parameters p. This will be used to fit for p
def objective_func(p, d, d_sig):
    sum_over = 0
    poly = timing_van_jax([p], 3)[0]

    #Loop over the bins
    i = 0
    while i < jnp.size(jnp.array(list(coeff.values())), axis=0):
        f_sig = jnp.sqrt(jnp.matmul(poly, jnp.matmul(cov[i], poly.T))) #Finding uncertainty of surrogate function at point p
        adj_res_sq = (d[i]-jnp.matmul(coeff[i], poly.T))**2/(d_sig[i]**2 + f_sig**2) #Inner part of summation
        sum_over = sum_over + adj_res_sq
        i += 1
    return sum_over

