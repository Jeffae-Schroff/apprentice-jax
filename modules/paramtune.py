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
            #TODO: waiting for harvey's h5
            self.target_binidns = jnp.array(range(self.fits.p_coeffs.shape[0]))
            self.target_values = [target_data[k][0] for k in target_data]
            self.target_error = [target_data[k][1] for k in target_data]
        
        self.obj_args = (self.target_values, self.target_error, jnp.take(self.fits.p_coeffs, self.target_binidns, axis = 0))
        if kwargs['covariance']:
            self.obj_args = self.obj_args + (jnp.take(self.fits.cov, self.target_binidns, axis = 0),)
            self.objective = self.objective_func
            self.objective_name = 'cov'
        else:
            self.objective = self.objective_func_no_err
            self.objective_name ='no_err'
        if type(initial_guess) == str:
            initial_guess = self.calculate_initial(initial_guess)
            print("Calculated inital guess: ", initial_guess) 
        self.p_opt = opt.minimize(self.objective, initial_guess, args = self.obj_args, method='TNC')

        # self.p_opt = opt.minimize(self.objective, initial_guess, bounds = [(1,2),(-1.2,-0.8)],
        # args = self.obj_args, method='TNC',tol=1e-6, options={'maxiter':1000, 'accuracy':1e-6})
        # temp to match apprentice.
        

        opt_obj = self.objective(self.p_opt.x, *self.obj_args)
        #Watch out for this if we have experiments with a significant amount of empty bins
        self.ndf = len(self.target_binidns) - self.fits.dim
        print("Tuned Parameters: ", self.p_opt.x, ", Objective = ", opt_obj, ", chi2/ndf = ", opt_obj/self.ndf)

    def graph_tune(self, obs_name, graph_file = None, new_figure = True):
        #only select binids from out obs_name for which there is target data
        obs_bin_idns = jnp.intersect1d(jnp.array(self.fits.index[obs_name]), self.target_binidns)
        poly_opt = timing_van_jax([self.p_opt.x], 3)[0]
        tuned_y = jnp.matmul(jnp.array([self.fits.p_coeffs[b] for b in obs_bin_idns]), poly_opt.T)
        
        if new_figure: plt.figure()
        plt.title("Placeholder")
        #Might be something like "number of events", but depends on what observable is, find in Harvey's h5 file
        plt.ylabel("Placeholder")
        plt.xlabel(obs_name + " bins")
        num_bins = len(obs_bin_idns)
        num_ticks = 7 if num_bins > 14 else num_bins
        plt.xticks([round(x/num_ticks) for x in range(0, num_bins*num_ticks, num_bins)]+[num_bins])
        edges = range(num_bins + 1)
        plt.stairs([self.target_values[b] for b in obs_bin_idns], edges, label = 'Target Data')
        plt.stairs(tuned_y, edges, label = 'Surrogate(Tuned Parameters)')
        
        plt.legend()
        if not graph_file == None: plt.savefig(graph_file)

    #TODO: Record param name(s) so this can take param names and visualize that param
    # It's in attributes of the param table of h5, so polyft needs to be changed too
    def graph_objective(self, dof_scale = 1, graph_file = None, new_figure = True):
        confidence_level = 0.68268949 #within 1 standard deviation
        edof = dof_scale*(self.ndf)
        target_dev = chi2.ppf(confidence_level, edof)
        print(f"target deviation {target_dev:.4f}, with confidence level {cl:.4f}, edof {edof:.4f}")
        minX, maxX = self.fits.X.min(axis = 0), self.fits.X.max(axis = 0)
        if new_figure: plt.figure()
        if self.fits.dim == 1:
            graph_density = 500
            x = jnp.arange(minX[0], maxX[0], (maxX[0]-minX[0])/graph_density)
            y = jnp.apply_along_axis(self.objective, 1, jnp.expand_dims(x, axis=1), *self.obj_args)
            p = plt.plot(x, y, label = self.objective_name)

            obj_opt = self.objective(self.p_opt.x, *self.obj_args)
            plt.plot(self.p_opt.x, obj_opt, color = p[-1].get_color(), marker = 'o')
            plt.axhline(target_dev + obj_opt, color = p[-1].get_color(), linestyle = 'dotted')
            within_error = jnp.where(y < target_dev + obj_opt)[0] # possibly vulnerable to local minima
            low_bound, high_bound = x[within_error[0]], x[within_error[-1]] 
            plt.plot([], [], color = p[-1].get_color(), linestyle = 'dotted', 
            label= "[{:.4f}, {:.4f}]".format(low_bound,high_bound))
            plt.legend()
            plt.title('Parameter regions within 1 std of tuned result')
            plt.ylabel('Objective')
            plt.xlabel('MPIalphaS') #TODO make automatic
            plt.yscale("log")
        else:
            print("not implemented")
        if not graph_file == None:
            plt.savefig(graph_file)

        

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
            #We don't want to divide by 0. For real data, the error might be zero because no events were observed in the bin.
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