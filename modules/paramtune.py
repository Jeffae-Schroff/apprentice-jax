import jax. random as random
import jax.numpy as jnp
from timing import timing_van_jax
import scipy.optimize as opt
from scipy.stats import chi2
import matplotlib.pyplot as plt
from modules.polyfit import Polyfit
import json
class Paramtune:
    '''
    TODO: document
    '''
    def __init__(self, npz_file, target_json, initial_guess = None, **kwargs):
        self.fits = Polyfit(npz_file, **kwargs)
        
        # Read in target data
        if 'target_bins' in kwargs.keys() and 'target_values' in kwargs.keys() and 'target_errors' in kwargs.keys():
            self.target_binidns = jnp.array([self.fits.bin_idn(b) for b in kwargs['target_bins']])
            self.target_values = kwargs['target_values']
            self.target_error = kwargs['target_errors']
        else:
            with open(target_json, 'r') as f:
                target_data = json.loads(f.read())
            #TODO: not generalized
            self.target_binidns = jnp.array(range(self.fits.p_coeffs.shape[0]))
            self.target_values = [target_data[k][0] for k in target_data]
            self.target_error = [target_data[k][1] for k in target_data]
        
        self.obj_args = (self.target_values, self.target_error, jnp.take(self.fits.p_coeffs, self.target_binidns, axis = 0))
        if kwargs['covariance']:
            self.obj_args = self.obj_args + (jnp.take(self.fits.cov, self.target_binidns, axis = 0),)
            self.objective = self.objective_func
        else:
            self.objective = self.objective_func_no_err
        if type(initial_guess) == str:
            initial_guess = self.calculate_initial(initial_guess)
            print("Calculated inital guess: ", initial_guess) 
        self.p_opt = opt.minimize(self.objective, initial_guess, args = self.obj_args, method='Nelder-Mead')
        print("Tuned Parameters: ", self.p_opt.x)

    def graph_tune(self, obs_name, graph_file = None):
        obs_bin_idns = self.fits.index[obs_name]
        poly_opt = timing_van_jax([self.p_opt.x], 3)[0]
        tuned_y = jnp.matmul(jnp.array([self.fits.p_coeffs[b] for b in obs_bin_idns]), poly_opt.T)
        
        plt.figure()
        plt.title("Placeholder")
        #Might be something like "number of events", but depends on what observable is, find in Harvey's h5 file
        plt.ylabel("Placeholder")
        plt.xlabel(obs_name + " bins")
        num_bins = len(self.fits.index[obs_name])
        num_ticks = 7
        plt.xticks([round(x/num_ticks) for x in range(0, num_bins*num_ticks, num_bins)]+[num_bins])
        edges = range(num_bins + 1)
        plt.stairs([self.target_values[b] for b in obs_bin_idns], edges, label = 'Target Data')
        plt.stairs(tuned_y, edges, label = 'Surrogate(Tuned Parameters)')
        
        plt.legend()
        if not graph_file == None:
            plt.savefig(graph_file)

    def graph_contour(self, obs_name, dof_scale = 1):
        obs_bin_idns = self.fits.obs_index[obs_name]
        target_deviation = chi2.ppf(0.68268949, dof_scale)
        #temp

    def calculate_initial(self, method):
        #takes guess in param range with smallest objective.
        if method == 'sample_range':
            num_samples = 50
            #TODO: replace 2403 before this gets used on Cori
            samples = random.uniform(random.PRNGKey(2403), (num_samples,self.fits.dim),\
                minval = self.fits.X.min(axis = 0), maxval = self.fits.X.max(axis = 0), dtype=jnp.float64)
            objective = jnp.apply_along_axis(self.objective, 1, samples, *self.obj_args)
            return samples[jnp.argmin(objective)]
        elif method == 'mc_runs':
            objective = jnp.apply_along_axis(self.objective, 1, self.fits.X, *self.obj_args)
            return self.fits.X[jnp.argmin(objective)]
        else:
            print("initial guess calulation method invalid")


    def objective_func(self, params, d, d_sig, coeff, cov):
        sum_over = 0
        poly = timing_van_jax([params], self.fits.order)[0]

        #Loop over the bins
        for i in self.target_binidns:
            #We don't want to divide by 0. For real data, the error might be zero because no events were observered in the bin.
            if d_sig[i] == 0.0:
                continue
            f_sig = jnp.sqrt(jnp.matmul(poly, jnp.matmul(cov[i], poly.T))) #Finding uncertainty of surrogate function at point p
            adj_res_sq = (d[i]-jnp.matmul(coeff[i], poly.T))**2/(d_sig[i]**2 + f_sig**2) #Inner part of summation
            sum_over = sum_over + adj_res_sq
        return sum_over

    def objective_func_no_err(self, p, d, d_sig, coeff):
        sum_over = 0
        poly = timing_van_jax([p], self.fits.order)[0]

        #Loop over the bins
        for i in range(jnp.size(jnp.array(coeff), axis=0)):
            if d_sig[i] == 0.0:
                continue
            adj_res_sq = (d[i]-jnp.matmul(coeff[i], poly.T))**2/(d_sig[i]**2) #Inner part of summation
            sum_over = sum_over + adj_res_sq
        return sum_over