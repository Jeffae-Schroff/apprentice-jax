import jax
import numpy as np
import jax.numpy as jnp
import scipy.optimize as opt
from scipy.stats import chi2
import matplotlib.pyplot as plt
from modules.polyfit import Polyfit
import json
import h5py

"""
        TODO: Convert documentation to actual doc using str.__doc__
"""

class Paramtune:
    """
    Outer loop optimization.

    Encompasses the outer loop optimization, which fits each bin's surrogate function (some
    polynomial in the parameters) to the target bin values by tuning the parameters.

    Attributes
    ----------
    fits : Polyfit object
        Inner-loop optimization surrogate functions. See Polyfit docu for detail
    target_binidns : JAX array
    target_values : JAX array
    target_error : JAX array
    obj_args :
    objective : Function
        The objective function which is used in optimization. Currently there is a version considering error and one which does not.
    objective_name : str
        Marker denoting which objective function is used.
    p_opt : 
        The output of the optimization. Use p_opt.x to obtain tuned parameters.
    cov : 
        Covariance matrix associated with the tuned parameters.


    Methods
    -------

    """

    def __init__(self, npz_file, target_json, initial_guess = 'sample_range', tune = True, **kwargs):
        self.fits = Polyfit(npz_file, **kwargs)
        self.tune = tune
        # Read in target data:
        if 'target_bins' in kwargs.keys() and 'target_values' in kwargs.keys() and 'target_errors' in kwargs.keys():
            self.target_binidns = jnp.array([self.fits.bin_idn(b) for b in kwargs['target_bins']]) #The user can select which bins they wish to be considered in the tuning
            print(self.target_binidns)
            self.target_values = kwargs['target_values']
            self.target_error = kwargs['target_errors']
        elif 'target_h5' in kwargs.keys() and 'target_binidns' in kwargs.keys():
            self.target_binidns = kwargs['target_binidns']
            f = h5py.File(kwargs['target_h5'], "r")
            self.target_values = jnp.array(f['main'][:,1], dtype=np.float64)[self.target_binidns]
            self.target_error = jnp.array(f['main'][:,7], dtype=np.float64)[self.target_binidns]

        else:
            # use all target data in json
            with open(target_json, 'r') as f:
                target_data = json.loads(f.read())
            self.target_binidns = jnp.array(range(self.fits.p_coeffs.shape[0]))
            self.target_values = jnp.array([target_data[k][0] for k in target_data])
            self.target_error = jnp.array([target_data[k][1] for k in target_data])
        
        # Filter target data. Perhaps expand this to more than value 0, error 0?
        valid = []
        for i, (value, error) in enumerate(zip(self.target_values, self.target_error)):
            if not (value == 0 and error == 0):
                valid.append(i)
        self.target_binidns = jnp.take(self.target_binidns, jnp.array(valid))
        self.target_values = jnp.take(self.target_values, jnp.array(valid))
        self.target_error = jnp.take(self.target_error, jnp.array(valid))

        self.obj_args = (self.target_values, self.target_error, jnp.take(self.fits.p_coeffs, self.target_binidns, axis = 0))
        if kwargs['covariance']:
            self.obj_args = self.obj_args + (jnp.take(self.fits.cov, self.target_binidns, axis = 0),)
            self.objective = self.objective_func
            self.objective_name = 'cov'
        else:
            self.objective = self.objective_func_no_err
            self.objective_name ='no_err'
        
        self.ndf = len(self.target_binidns) - self.fits.dim
        if tune:
            if type(initial_guess) == str:
                initial_guess = self.calculate_initial(initial_guess)
                print("Calculated inital guess: ", initial_guess) 
            self.p_opt = opt.minimize(self.objective, initial_guess, args = self.obj_args, method='TNC')

            # self.p_opt = opt.minimize(self.objective, initial_guess, bounds = [(1,2),(-1.2,-0.8)],
            # args = self.obj_args, method='TNC',tol=1e-6, options={'maxiter':1000, 'accuracy':1e-6})
            # temp to match apprentice.
            

            opt_obj = self.objective(self.p_opt.x, *self.obj_args)
            #Watch out for this if we have experiments with a significant amount of empty bins
            print("Tuned Parameters: ", self.p_opt.x, ", Objective = ", opt_obj, ", chi2/ndf = ", opt_obj/self.ndf)

            #Calculating covariance of parameters by means of inverse Hessian
            coeff_target_bins = jnp.take(self.fits.p_coeffs, self.target_binidns, axis = 0)
            def res_sq(param):
                poly = jnp.asarray(self.fits.vandermonde_jax([param], self.fits.order)[0])
                return jnp.sum(jnp.square(coeff_target_bins@poly - jnp.asarray(self.target_values)))
            def Hessian(func):
                return jax.jacfwd(jax.jacrev(res_sq))
            cov = jnp.linalg.inv(Hessian(res_sq)(self.p_opt.x))
            fac = res_sq(self.p_opt.x)/(len(self.target_values) - len(self.p_opt.x))
            self.cov = cov*fac
            print("Covariance of Tuned Parameters: ", cov*fac)

    def calculate_initial(self, method):
        #takes guess in param range with smallest objective.
        if method == 'sample_range':
            num_samples = 50
            #TODO: make sure 2043 seed okay before this goes to Cori
            samples = jax.random.uniform(jax.random.PRNGKey(2403), (num_samples,self.fits.dim),\
                minval = self.fits.X.min(axis = 0), maxval = self.fits.X.max(axis = 0), dtype=jnp.float64)
            objective = jnp.apply_along_axis(self.objective, 1, samples, *self.obj_args)
            return samples[jnp.argmin(objective)]
        elif method == 'mc_runs':
            objective = jnp.apply_along_axis(self.objective, 1, self.fits.X, *self.obj_args)
            return self.fits.X[jnp.argmin(objective)]
        else:
            print("initial guess calulation method invalid")

    #Objective function which considers the errors in the inner-loop coefficients
    def objective_func(self, params, d, d_sig, coeff, cov):
        sum_over = 0
        poly = self.fits.vandermonde_jax([params], self.fits.order)[0]

        #Loop over the bins
        for i in self.target_binidns:
            f_sig = jnp.sqrt(jnp.matmul(poly, jnp.matmul(cov[i], poly.T))) #Finding uncertainty of surrogate function at point p
            adj_res_sq = (d[i]-jnp.matmul(coeff[i], poly.T))**2/(d_sig[i]**2 + f_sig**2) #Inner part of summation
            sum_over = sum_over + adj_res_sq
        return sum_over

    #Objective function which does not consider the errors in the inner-loop coefficients
    def objective_func_no_err(self, p, d, d_sig, coeff):
        sum_over = 0
        poly = self.fits.vandermonde_jax([p], self.fits.order)[0]

        #Loop over the bins
        for i in range(jnp.size(jnp.array(coeff), axis=0)):
            adj_res_sq = (d[i]-jnp.matmul(coeff[i], poly.T))**2/(d_sig[i]**2) #Inner part of summation
            sum_over = sum_over + adj_res_sq
        return sum_over
    
    # TODO: parallelize. Does it make sense for this to be attached to a specific paramtune object? 
    def graph_chi2_sample(self, input_h5, num_samples, sample_prop, color, new_figure = True, save_figure = None, graph_range = [0.3,30], save_file = None):
        num_runs = self.fits.X.shape[0]
        sample_size = round(sample_prop * num_runs)
        print("Tuning with", num_samples, "samples of size", sample_size, "out of", num_runs)

        og_fits = self.fits
        #use one initial guess for all tunes
        initial_guess = self.calculate_initial('sample_range')
        #make many Polyfits calculated with sampled mc_runs
        p_opt, chi2ndf = [],[]
        for i in range(num_samples):
            #temporarily replace self.fits with a Polyfit made with mc_run sample.
            self.fits = Polyfit(None, sample_size, False, input_h5 = input_h5,
            order = self.fits.order, covariance = self.fits.has_cov)
            #set up objective arguments using sampled Polyfit
            obj_args = (self.target_values, self.target_error, jnp.take(jnp.array(self.fits.p_coeffs), self.target_binidns, axis = 0))
            if self.fits.has_cov:
                obj_args = obj_args + (jnp.take(jnp.array(self.fits.cov), self.target_binidns, axis = 0),)
            #do tuning
            p_opt.append(opt.minimize(self.objective, initial_guess, args = obj_args, method='TNC')) #only saves tuned parans
            chi2ndf.append((self.objective(p_opt[i].x, *obj_args)/self.ndf).tolist()) #ndf unchanged by using sample
            print("Sample", i, "Tuned:", p_opt[i].x, "| chi2/ndf:", chi2ndf[i])
        self.fits = og_fits
        
        if save_file:
            results = {}
            # results['chi2ndf'] = [c.tolist() for c in chi2ndf]
            results['chi2ndf'] = chi2ndf
            results['tuned_p'] = [list(p.x) for p in p_opt]
            # json_results = json.dumps(results)
            with open(save_file, "w") as f:
                json.dump(results, f, indent=4)

        if new_figure: 
            plt.figure()
            plt.title("chi2/ndf for " + str(num_samples) + " mc_run samples")
            plt.ylabel("Frequency")
            plt.xlabel("chi/ndf")
            plt.xscale('log')       
        label = self.objective_name + ": {:.2f} +/- {:.2f}".format(jnp.mean(jnp.array(chi2ndf)), jnp.std(jnp.array(chi2ndf)))
        plt.hist(chi2ndf, bins = 'doane', label = label, range = graph_range, facecolor = color)
        plt.legend()
        if save_figure:
            plt.savefig(save_figure)


    #TODO: Record param name(s) so this can take param name(s) and visualize that param
    # It's in attributes of the param table of h5, so polyfit needs to be changed too
    def graph_objective(self, dof_scale = 1, graph_file = None, new_figure = True, graph_range = None):
        std = 1
        confidence_level = 0.68268949 * std #within 1 standard deviation
        edof = dof_scale*(self.ndf)
        target_dev = chi2.ppf(confidence_level, edof)
        print(f"target deviation {target_dev:.4f}, with confidence level {confidence_level:.4f}, edof {edof:.4f}")
        minX, maxX = self.fits.X.min(axis = 0), self.fits.X.max(axis = 0)
        if new_figure: plt.figure()
        if self.fits.dim == 1:
            graph_density = 1000
            if graph_range is None:
                x = jnp.arange(minX[0], maxX[0], (maxX[0]-minX[0])/graph_density)
            else:
                x = jnp.arange(graph_range[0], graph_range[1], (graph_range[1]-graph_range[0])/graph_density)

            objective_x = jnp.apply_along_axis(self.objective, 1, jnp.expand_dims(x, axis=1), *self.obj_args)
            if self.tune:
                objective_opt = self.objective(self.p_opt.x, *self.obj_args)
                y = objective_x - objective_opt
                p = plt.plot(x, y, label = self.objective_name)
                plt.plot(self.p_opt.x, 0, color = p[-1].get_color(), marker = 'o', markersize=4)
            else:
                y = objective_x 
                p = plt.plot(x, y, label = self.objective_name)

            

           
            plt.axhline(target_dev, color = p[-1].get_color(), linestyle = 'dotted')
            within_error = jnp.where(y < target_dev)[0] # possibly vulnerable to local minima
            low_bound, high_bound = x[within_error[0]], x[within_error[-1]] 
            plt.plot([], [], color = p[-1].get_color(), linestyle = 'dotted', 
            label= "[{:.4f}, {:.4f}]".format(low_bound,high_bound))
            plt.legend()
            plt.title('Parameter regions within ' + str(std) + ' std of tuned result')
            plt.ylabel('Objective - Optimal objective')
            plt.xlabel('MPIalphaS ' "[{:.4f}, {:.4f}]".format(minX[0], maxX[0])) #TODO make automatic
            #use logscale if the graph is spiky
            if(max(y) > target_dev*50): 
                plt.yscale("log")
        else:
            print("not implemented")
        if not graph_file == None:
            plt.savefig(graph_file)

    def graph_tune(self, obs_name, graph_file = None):
        #only select binids from obs_name for which there is target data
        obs_bin_idns = jnp.intersect1d(self.target_binidns, jnp.array(self.fits.index[obs_name]))
        poly_opt = self.fits.vandermonde_jax([self.p_opt.x], 3)[0]
        tuned_y = jnp.matmul(jnp.array([self.fits.p_coeffs[b] for b in obs_bin_idns]), poly_opt.T)
        plt.figure()
        plt.title("Placeholder")
        #Might be something like "number of events", but depends on what observable is, find in Harvey's h5 file
        plt.ylabel("Placeholder")
        plt.xlabel(obs_name + " bins")
        num_bins = len(obs_bin_idns)
        num_ticks = 7 if num_bins > 14 else num_bins #make whole numbers
        plt.xticks([round(x/num_ticks) for x in range(0, num_bins*num_ticks, num_bins)]+[num_bins])
        edges = range(num_bins + 1)
        plt.stairs([self.target_values[b] for b in obs_bin_idns], edges, label = 'Target Data')
        plt.stairs(tuned_y, edges, label = 'Surrogate(Tuned Parameters)')
        
        plt.legend()
        if not graph_file == None: plt.savefig(graph_file)
    
    # probably don't call if there are more than 20 observables w/ target data
    def graph_envelope_target(self):
        for obs_name in self.fits.obs_index.keys():
            obs_bin_idns = jnp.intersect1d(self.target_binidns, jnp.array(self.fits.index[obs_name]))
            if obs_bin_idns.size > 0:
                self.fits.graph_envelope([obs_name])
                plt.stairs([self.target_values[b] for b in obs_bin_idns], range(len(obs_bin_idns) + 1), label = 'target')
                plt.legend()