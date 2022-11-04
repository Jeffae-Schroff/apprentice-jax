import jax.numpy as jnp
from timing import timing_van_jax
import scipy.optimize as opt
import matplotlib.pyplot as plt
from modules.polyfit import Polyfit
import json
class Paramtune:
    def __init__(self, target_json, p_coeffs_npz, chi2res_npz, obs_sample = None, initial_guess = None, **kwargs):
        self.fits = Polyfit(p_coeffs_npz, chi2res_npz, **kwargs)
        
        # Read in target data
        with open(target_json, 'r') as f:
            target_data = json.loads(f.read())
        # stripping '/' again to match polyfit
        self.target_values = {k.replace('/', ''): target_data[k][0] for k in target_data}
        self.target_error = {k.replace('/', ''): target_data[k][1] for k in target_data}
        if not obs_sample is None:
            self.target_values = {k:self.target_values[k] for k in list(self.target_values.keys())[:obs_sample]}
            self.target_error = {k:self.target_error[k] for k in list(self.target_error.keys())[:obs_sample]}
        
        args = (list(self.target_values.values()), list(self.target_error.values()), list(self.fits.p_coeffs.values()))
        if ('cov_npz' in kwargs.keys()):
            args = args + (list(self.fits.cov.values()),)
            self.objective = self.objective_func
        else:
            self.objective = self.objective_func_no_err
        self.p_opt = opt.minimize(self.objective, initial_guess, args = args, method='Nelder-Mead')
        print("Tuned Parameters: ", self.p_opt.x)

    def graph_tune(self, obs_name, graph_file = None):
        bin_ids = self.fits.obs_index[obs_name]
        poly_opt = timing_van_jax([self.p_opt.x], 3)[0]
        tuned_y = jnp.matmul(jnp.array([self.fits.p_coeffs[b] for b in bin_ids]), poly_opt.T)
        
        plt.figure()
        plt.title("Placeholder")
        #Might be something like "number of events", but depends on what observable is, find in Harvey's h5 file
        plt.ylabel("Placeholder")
        plt.xlabel(obs_name + " bins")
        num_bins = len(self.fits.obs_index[obs_name])
        num_ticks = 7
        plt.xticks([round(x/num_ticks) for x in range(0, num_bins*num_ticks, num_bins)]+[num_bins])
        edges = range(num_bins + 1)
        plt.stairs([self.target_values[b] for b in bin_ids], edges, label = 'Target Data')
        plt.stairs(tuned_y, edges, label = 'Surrogate(Tuned Parameters)')
        
        plt.legend()
        if not graph_file == None:
            plt.savefig(graph_file)

    def graph_contour(self, obs_name):
        bin_ids = self.fits.obs_index[obs_name]
        #temp

    def objective_func(self, params, d, d_sig, coeff, cov):
        sum_over = 0
        poly = timing_van_jax([params], 3)[0]

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