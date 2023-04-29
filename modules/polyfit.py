import random
import h5py
from matplotlib import pyplot as plt
import scipy.optimize as opt
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from jax.config import config
jax.config.update("jax_enable_x64", True)



"""

"""

class Polyfit:
    """
    Inner loop optimization.

    Encompasses the inner loop optimization, which approximates each bin's dependence on the
    parameters with a polynomial of predefined order. Fits Monte Carlo generated data to
    polynomials by varying the coefficients of said polynomial.

    Attributes
    ----------
    input_h5 : str
        Filepath to h5 file containing data produced by Monte Carlo runs
    order: int
        Order of polynomial used to fit
    dim: int
    num_coeffs: int
    bin_ids: [str]
        Array of keys denoting each bin
    index:
    obs_index:
    p_coeffs: [[float]]
        2D array containing fitted polynomial coefficients for each bin. Terms in graded lexicographic order.
    cov: [[[float]]]
        3D array containing covariance matrix for fitted coefficients for each bin. Calculated via inverse Hessian
    X:
    Y:
    res: [float]
        Array of sumSq of residuals for each bin
    chi2ndf:
        TODO: fill this in

    Methods
    TODO
    -------

    """


    def __init__(self, npz_file, sample = None, **kwargs):
        """ 
        If 'input_h5' are 'order' are given, fit polynomials of order order to each bin in input_h5,
        
        If 'input_h5' and 'order' are not given,
        Load object from npz file

        Mandatory Arguments:
        npz_file -- filepath for npz file storing results. None->don't save
        Optional Arguments:
        sample -- if a number, sample this many mc runs to use in fit.
        Keyword Arguments:
        covariance -- whether or not to create covariance matrix. permanent for this object
        Optional Keyword Arguments (use to create fits):
        input_h5 -- the name of an h5 file with MC run results
        order -- the order of polynomial to fit each bin with
        reg_mode -- regression mode, default to lst_sq
        pdf_uncertainty -- quadratically add pdf uncertainty to error
        """

        if 'input_h5' in kwargs.keys() and 'order' in kwargs.keys() and 'covariance' in kwargs.keys():
            if 'cpu' in kwargs.keys() and kwargs['cpu']:
                config.update('jax_platform_name', 'cpu')
            self.input_h5 = kwargs['input_h5']
            self.order = kwargs['order']
            self.has_cov = kwargs['covariance']
            self.reg_mode = 'lst_sq'
            self.reg_param = 0
            self.objective_func = self.lst_sq
            if 'reg_mode' in kwargs.keys():
                self.reg_mode = kwargs['reg_mode']
                self.reg_param = kwargs['reg_param']
            

            f = h5py.File(self.input_h5, "r")
            self.X = jnp.array(f['params'][:], dtype=jnp.float64) #num_mc_runs * dim array
            self.param_names = np.array(f['params'].attrs['names']).astype(str) #apprentice stored it this way after merging yoda files
            self.Y = jnp.array(f['values'][:], dtype=jnp.float64) #num_bins * num_mc_runs array
            self.Y_err = jnp.array(f['total_err'][:], dtype=jnp.float64)
            self.bin_ids = np.array([x.decode() for x in f.get("index")[:]])
            self.obs_weights = jnp.ones(jnp.shape(self.bin_ids))
            self.original_dim = len(self.bin_ids)
            self.dim = self.X.shape[1]
            self.num_coeffs = self.numCoeffsPoly(self.dim, self.order) 

            #filter out bins with invalid value (>1000) for any mc_run
            invalid = jnp.array(jnp.any((abs(self.Y[:,:]) > 1000), axis = 1).nonzero()[0])
            if len(invalid) > 0:
                print("Filtered", len(invalid), "of", len(self.bin_ids), "total bins for invalid input")
            self.Y = jnp.delete(self.Y, invalid, axis=0)
            self.Y_err = jnp.delete(self.Y_err, invalid, axis=0)
            self.bin_ids = np.delete(self.bin_ids, invalid)
            self.obs_weights = jnp.delete(self.obs_weights, invalid)

            # If fit_obs is in arugments, it is a list of which observables to use (by order in the input file)
            #Replaced by select_obs, can delete?
            if 'fit_obs' in kwargs.keys():
                #Won't truncate e.g. d100, but will add leading zeroes to single digits to match pythia (?) format
                fit_obs = ['d' + str(obs).zfill(2) + '-' for obs in kwargs['fit_obs']]
                print("Choosing to fit observables with", fit_obs)

                #filter out bins with name that does not match fit_obs
                invalid = jnp.array([i for i, bin in enumerate(self.bin_ids) if not any([obs in bin for obs in fit_obs])])
                self.Y = jnp.delete(self.Y, invalid, axis=0)
                self.Y_err = jnp.delete(self.Y_err, invalid, axis=0)
                self.bin_ids = np.delete(self.bin_ids, invalid)
                self.obs_weights = jnp.delete(self.obs_weights, invalid)

            if 'num_bins' in kwargs.keys():
                #Won't truncate e.g. d100, but will add leading zeroes to single digits to match pythia (?) format
                num_bins = kwargs['num_bins']
                print("Selecting ", num_bins, " bins for fitting.")

                if num_bins < self.bin_ids.size: 
                    self.Y = self.Y[0:num_bins]
                    self.Y_err = self.Y_err[0:num_bins]
                    self.bin_ids = self.bin_ids[0:num_bins]

            #Only fit observables from given list
            if 'select_obs' in kwargs.keys():
                obs_list = kwargs['select_obs']
                id_cut = np.array([sb[0] for sb in np.char.rsplit(self.bin_ids, "#", maxsplit=1)])  #Removes bin numbers from index, leaving only a list of observable names
                inv = np.array([i for i, string in enumerate(id_cut) if not string in obs_list])
                print("Fitting observables ", obs_list)

                self.Y = jnp.delete(self.Y, inv, axis=0)
                self.Y_err = jnp.delete(self.Y_err, inv, axis=0)
                self.bin_ids = np.delete(self.bin_ids, inv)
                id_cut = np.delete(id_cut, inv)
                self.obs_weights = jnp.delete(self.obs_weights, inv)

                inv = np.array([])
                #Selecting bin_num bins from each observable (Only used to reduce number of bins for low-performance testing)
                if('bin_num' in kwargs.keys()):
                    obs_bin_num = kwargs['bin_num']
                    obs_name, counts = np.unique(id_cut, return_counts=True)

                    to_clear = counts - obs_bin_num
                    clear_dict = dict(zip(obs_name, to_clear))

                    id_cut_rev = np.flip(id_cut)

                    inv = np.array([])
                    for i, string in enumerate(id_cut_rev):
                        if clear_dict[string] > 0:
                            inv = np.append(inv, i)
                            clear_dict[string] -= 1

                    inv = np.flip(np.array([np.size(id_cut_rev) - 1 - i for i in inv])).astype(int)
                    self.Y = jnp.delete(self.Y, inv, axis=0)
                    self.Y_err = jnp.delete(self.Y_err, inv, axis=0)
                    self.bin_ids = np.delete(self.bin_ids, inv)
                    id_cut = np.delete(id_cut, inv)
                    self.obs_weights = jnp.delete(self.obs_weights, inv)

                #Defines weights for each observable, will affect how heavily they are considered in outer-loop optimization
                if('bin_weights' in kwargs.keys()):
                    weight_list = kwargs['bin_weights']
                    for i, weight in enumerate(weight_list):
                        for j, cut_id in enumerate(id_cut):
                            if cut_id == obs_list[i]:
                                self.obs_weights = self.obs_weights.at[j].set(weight)

            
            #if mc_target, remove a column from the fit data for new experiment 
            self.mc_target_X = None
            self.mc_target = None
            self.mc_target_err = None
            if 'mc_target' in kwargs.keys():
                invalid = kwargs['mc_target']
                self.mc_target_X = self.X[invalid]
                self.mc_target = self.Y[:, invalid]
                self.mc_target_err = self.Y_err[:, invalid]
                print("mc_target: ", invalid, " param value: ", self.mc_target_X)
                self.X = jnp.delete(self.X, invalid, axis=0)
                self.Y = jnp.delete(self.Y, invalid, axis=1)
                self.Y_err = jnp.delete(self.Y_err, invalid, axis=1)
            
            #sample mc_runs. Number of used bins is consistent (we just filtered bins).
            if not sample is None:                                         #(num MC runs)
                if type(sample) is int and sample > self.dim and sample <= self.X.shape[0]:
                    #TODO: Cori-proof
                    random.seed()
                    sample_which = jnp.array(random.sample(range(self.X.shape[0]), sample))
                    self.X = jnp.take(self.X, sample_which, axis=0)
                    self.Y = jnp.take(self.Y, sample_which, axis=1)
                else:
                    print("invalid sample input")
            

            # the index keys bin names to the array indexes in f.get(index) with binids matching that bin name
            self.index = {}
            # If bin not a key in index yet, start a new list as its value. Append count to bin's value. 
            [self.index.setdefault(bin.split('#')[0], []).append(count) for count,bin in enumerate(self.bin_ids)]  
            # keys observable names to their error names
            self.obs_index = {}
            [self.obs_index.setdefault(bin_name.split('[')[0], []).append(bin_name) if ('[') in bin_name 
                else self.obs_index.setdefault(bin_name, []) for bin_name in self.index.keys()]
            
            # add pdf_uncertainty to y_err
            if 'pdf_uncertainty' in kwargs.keys() and kwargs['pdf_uncertainty']:
                for bin_id in self.bin_ids:
                    obs_name, bin_num = bin_id.split('#')[0], bin_id.split('#')[1]
                    if obs_name in self.obs_index:
                        bin_Y = self.Y[self.bin_idn(bin_id),:]
                        bin_Y_err = self.Y_err[self.bin_idn(bin_id),:]
                        pdf_ids = [obs_name + '#' + bin_num for obs_name in self.obs_index[obs_name] if ':pdf:' in obs_name]
                        pdf_err = jnp.array([abs(bin_Y - self.Y[self.bin_idn(p),:]) for p in pdf_ids])
                        pdf_err = jnp.max(pdf_err, axis=0)
                        if(max(pdf_err) > 0):
                            print("pdf: ", pdf_err)
                        self.Y_err = self.Y_err.at[self.bin_idn(bin_id),:].set(jnp.sqrt(jnp.square(bin_Y_err) + jnp.square(pdf_err)))

            #optimize this loop later
            #we do want to fit curves to every bin name (values and uncertainties) for w_err
            #TODO: consider use for asymmetric uncertainty
            VM = self.vandermonde_jax(self.X, self.order)
            self.p_coeffs, self.p_coeffs_err, self.chi2ndf, self.res = [],[],[],[]
            if self.has_cov: self.cov = []


            if self.Y.shape[0]/self.original_dim <= 0.005:
                print("Only attempting to fit ", self.Y.shape[0], " out of ", self.original_dim, " bins.")
            
            self.skip_idn = []
            skip_list = open("skipped_bins.txt", "w")
            skip_count = 0
            for bin_count, bin_id in enumerate(self.bin_ids):
                print("\rAttempting to fit {:d} of {:d}: {:60s}".format(bin_count + 1, self.Y.shape[0], bin_id), end='')
                
                bin_Y = self.Y[self.bin_idn(bin_id),:]
                bin_Y_err = self.Y_err[self.bin_idn(bin_id),:]
                bin_Y_err_is_zero = jnp.array([bin_Y_err == 0])
                #Skip empty bins
                if not jnp.any(bin_Y) or not jnp.any(bin_Y_err) or VM.shape[0] - jnp.sum(bin_Y_err_is_zero) <= VM.shape[1]:
                    print("\nBin ", bin_count, " identically zero or has too many zero error across all runs, skipping!")
                    skip_list.write(bin_id + "\n")
                    self.skip_idn.append(bin_count)
                    skip_count += 1
                    self.p_coeffs.append(jnp.zeros((VM.shape[1],), dtype=jnp.float32).tolist())
                    self.res.append(0) #bin_res comes out of lstsq as a list
                    self.chi2ndf.append(0) #because it's supposed to be /ndf
                    if self.has_cov:
                        self.cov.append(jnp.zeros((VM.shape[1],VM.shape[1]), dtype=jnp.float32))
                    continue
                
                #Skip runs with 0 systematic error
                bin_Y = bin_Y[bin_Y_err != 0]
                bin_VM = VM[bin_Y_err != 0]
                bin_Y_err_adj = bin_Y_err[bin_Y_err != 0]
                
                #polynomialapproximation.coeffsolve2 code
                
            
                if self.reg_mode == 'ridge':
                    self.objective_func = self.ridge_obj
                    obj_args = (bin_Y, bin_VM, self.reg_param)
                elif self.reg_mode == 'ridge_w':
                    self.objective_func = self.ridge_obj_w
                    obj_args = (bin_Y, bin_Y_err_adj, bin_VM, self.reg_param)
                elif self.reg_mode == 'lasso':
                    self.objective_func = self.lasso_obj
                    obj_args = (bin_Y, bin_VM, self.reg_param)
                elif self.reg_mode == 'lasso_w':
                    self.objective_func = self.lasso_obj_w
                    obj_args = (bin_Y, bin_Y_err_adj, bin_VM, self.reg_param)
                elif self.reg_mode == 'lst_sq_w':
                    self.objective_func = self.lst_sq_w
                    obj_args = (bin_Y, bin_Y_err_adj, bin_VM)
                else:
                    obj_args = (bin_Y, bin_VM)

                guess = jnp.zeros((bin_VM.shape[1],), dtype=jnp.float32)
                c_opt = opt.minimize(self.objective_func, guess, args=obj_args, method='Nelder-Mead')
                bin_p_coeffs = c_opt.x
                bin_res = [jnp.sum(jnp.square(bin_Y-bin_VM@bin_p_coeffs))]

                surrogate_Y = self.surrogate(self.X, bin_p_coeffs)[bin_Y_err != 0]
                bin_chi2 = jnp.sum(jnp.divide(jnp.power((bin_Y - surrogate_Y), 2), surrogate_Y))

                self.p_coeffs.append(bin_p_coeffs.tolist())
                self.res.append(bin_res[0]) #bin_res comes out of lstsq as a list
                self.chi2ndf.append(bin_chi2/(self.num_coeffs-1)) #because it's supposed to be /ndf
                
                #Calculating covariance of coefficients using inverse Hessian
                def mini_obj(coeff):
                    return self.objective_func(coeff, *obj_args)
                def Hessian(func):
                    return jax.jacfwd(jax.jacrev(func))
                #polynomialapproximation.fit code
                if self.has_cov:
                    pcov = jnp.linalg.inv(Hessian(mini_obj)(bin_p_coeffs))
                    fac = bin_res[0] / (bin_VM.shape[0]-bin_VM.shape[1])
                    self.cov.append(pcov*fac)

                    # #Old code!
                    """cov = np.linalg.inv(VM.T@VM)
                    fac = bin_res / (VM.shape[0]-VM.shape[1])"""
                if jnp.any(jnp.isinf(pcov)):
                    print(bin_id)
            print("\n", skip_count, " bins skipped for zeros.")        
            print("\nFits written to", npz_file)
            if npz_file is not None:
                self.save(npz_file)

        #Initialize from npz file
        elif 'covariance' in kwargs.keys():
            print("loading ", npz_file)
            self.has_cov = kwargs['covariance']
            self.merge(npz_file, new = True)
        else:
            print('invalid args given to polyfit')

    def merge(self, all_npz, new = False):
        """ Merge new data to existing class variables.
        For now, if new data has same bin ids as old data, both copies are kept.
        """
        if all_npz is None:
             all_dict = {}
        else:
            all_dict = np.load(all_npz, allow_pickle=True)
            if new:
                self.order, self.dim, self.param_names = all_dict['order'], all_dict['dim'], all_dict['param_names']
            elif self.order != all_dict['order'] or self.dim != all_dict['dim'] or self.param_names != all_dict['param_names']:
                print("merging data with different order/dim/param_names is not allowed(error)")
            self.num_coeffs = self.numCoeffsPoly(self.dim, self.order)
            self.skip_idn = all_dict['skip_idn']

            jnp_vars = ['p_coeffs', 'chi2ndf', 'res', 'X', 'Y', 'Y_err', 'obs_weights']
            if self.has_cov: jnp_vars.append('cov')
            if 'mc_target' in all_dict.keys(): jnp_vars.extend(['mc_target_X', 'mc_target', 'mc_target_err'])
            for string in jnp_vars: #jnp: numbers only
                if new:
                    setattr(self, string, jnp.array(all_dict[string]))
                else:
                    setattr(self, string, jnp.concatenate([getattr(self, string), all_dict[string]]))
            
        # We recalculate index, obs_index from bin_ids. I decided to just store bin_ids because 
        # 1. npz likes np lists only and
        # Actually though we might be able to concatenate on merge if we store it. worth thinking about
        self.bin_ids = all_dict['bin_ids'] if new else jnp.concatenate([self.bin_ids, all_dict['bin_ids']])
        self.index = {}
        [self.index.setdefault(bin.split('#')[0], []).append(count) for count,bin in enumerate(self.bin_ids)]
        self.obs_index = {}
        [self.obs_index.setdefault(bin_name.split('[')[0], []).append(bin_name) if ('[') in bin_name 
            else self.obs_index.setdefault(bin_name, []) for bin_name in self.index.keys()]


    def save(self, all_npz):
        """ Save data to given filepath

        Mandatory arguments:
        all_npz -- filepath for npz file of data
        """
        all_dict = {}
        all_vars = ['p_coeffs', 'chi2ndf', 'res', 'X', 'Y', 'Y_err', 'bin_ids', 'obs_weights', 'dim', 'order', 'skip_idn', 'param_names']
        if self.has_cov: all_vars.append('cov') 
        if not self.mc_target is None: all_vars.extend(['mc_target_X', 'mc_target', 'mc_target_err'])
        for string in all_vars:
            all_dict[string] = getattr(self, string)
        np.savez(all_npz, **all_dict)

    def graph_bin(self, bin_id):
        """
        Takes bin_id, in format bin_name#bin_number
        graphs values of bin across runs and the surrogate function that was fitted to it
        """
        plt.figure()
        plt.title("Verification graph for " + bin_id)
        plt.ylabel("Values")
        plt.xlabel("MC runs")
        edges = range(self.X.shape[0] + 1)
        plt.stairs([y for y in self.Y[self.bin_idn(bin_id), :]], edges, label = 'Target Data')
        surrogate, chi2, res, cov = self.get_surrogate_func(bin_id)
        surrogate_y = surrogate(self.X)
        plt.stairs(surrogate_y, edges, label = 'Surrogate(Tuned Parameters)')
        plt.legend()

    def graph_envelope(self, graph_obs = None):
        ymin = self.Y.min(axis=1)
        ymax = self.Y.max(axis=1)
        if graph_obs is None:
            graph_obs = self.obs_index.keys()
        elif type(graph_obs) == str:
            graph_obs = [graph_obs]
        for obs in graph_obs:
            obs_bin_idns = jnp.array(self.index[obs])
            plt.figure()
            plt.title("Envelope of " + obs)
            #Might be something like "number of events", but depends on what observable is.
            plt.ylabel("Placeholder")
            plt.xlabel(obs + " bins")
            num_bins = len(obs_bin_idns)
            num_ticks = 7 if num_bins > 14 else num_bins #make whole numbers
            edges = range(num_bins + 1)
            plt.xticks([round(x/num_ticks) for x in range(0, num_bins*num_ticks, num_bins)]+[num_bins])
            plt.stairs(jnp.take(ymin, obs_bin_idns), edges, label = 'min')
            plt.stairs(jnp.take(ymax, obs_bin_idns), edges, label = 'max')
            plt.legend()
            

    #METHODS RELATING TO SURROGATE FUNCTION

    def get_surrogate_func(self, bin_id):
        """
        Takes bin_id, either in format bin_name#bin_number, or int: place of the bin in pceoff etc.
        Returns a surrogate that does not need pceoffs as an input; just takes x(param values).
        Returns chi2/ndf of fit, residual(s), and cov if it exists.
        """
        bin_idn = self.bin_idn(bin_id) if type(bin_id) is str else bin_id     
        return partial(self.surrogate, p_coeffs = self.p_coeffs[bin_idn]), self.chi2ndf[bin_idn], self.res[bin_idn], \
            self.cov[bin_idn] if self.has_cov else None
    
    def surrogate(self, x, p_coeffs):
        """
        Takes a list/array of param sets, and returns the surrogate's estimate for their nominal values.
        The length of the highest dimension of the list/array is the number of params.
        """
        x = jnp.array(x)
        dim = x.shape[-1]
        #has to be mutable the way this is written
        term_powers = jnp.zeros(dim)
        pvalues = []
        last_axis = None if len(x.shape) == 1 else len(x.shape)-1
        for pcoeff in p_coeffs:
            #appending the value of the term for each param set given
            pvalues.append(pcoeff*jnp.prod(x**term_powers, axis = last_axis))
            term_powers = self.mono_next_grlex(dim, term_powers)
        #add along first axis to sum the terms for each param set
        return jnp.sum(jnp.asarray(pvalues), axis=None if len(x.shape) == 1 else 0)
    
    def vandermonde_jax(self, params, order):
        """
        Construct the Vandermonde matrix.
        If params is a 2-D array, the highest dimension indicates number of parameters.
        """
        try:
            dim = len(params[0])
        except:
            dim = 1
        params = jnp.array(params)

        #Generate all compositions, will become parameter exponents. We will take params to the power of grlex_pow element-wise
        if dim == 1:
            grlex_pow = jnp.array(range(order+1))
        else:
            term_list = [[0]*dim]
            for i in range(1, self.numCoeffsPoly(dim, order)):
                term_list.append(self.mono_next_grlex(dim, jnp.asarray(term_list[-1][:])))
            grlex_pow = jnp.array(term_list)
        
        #Take each parameter to powers in grlex_pow term by term
        if dim == 1:
            V = jnp.zeros((len(params), self.numCoeffsPoly(dim, order)), dtype=jnp.float64)
            for a, p in enumerate(params): 
                V = V.at[a].set(p**grlex_pow)
            return V
        else:
            V = jnp.power(params, grlex_pow[:, jnp.newaxis])
            return jnp.prod(V, axis=2).T

    def bin_idn(self, bin_id):
        # Convert bin_id in format bin_name#bin_number to binidn
        bin_name = bin_id.split('#')[0]
        bin_number = int(bin_id.split('#')[1])
        if bin_name in self.index.keys() and len(self.index[bin_name]) > bin_number:
            return self.index[bin_name][bin_number]
        else:
            return -1
    
    def numCoeffsPoly(self, dim, order):
        """
        Number of coefficients a dim-dimensional polynomial of order order has (C(dim+order, dim)).
        """
        ntok = 1
        r = min(order, dim)
        for i in range(r):
            ntok = ntok * (dim + order - i) / (i + 1)
        return int(ntok)
    def mono_next_grlex(self, m, x):
        #  Author:
        #
        #    John Burkardt
        #
        #     TODO --- figure out the licensing thing https://people.sc.fsu.edu/~jburkardt/py_src/monomial/monomial.html

        #  Find I, the index of the rightmost nonzero entry of X.
        i = 0
        for j in range(m, 0, -1):
            if 0 < x[j-1]:
                i = j
                break

        #  set T = X(I)
        #  set X(I) to zero,
        #  increase X(I-1) by 1,
        #  increment X(M) by T-1.
        if i == 0:
            x = x.at[m-1].set(1)
            return x
        elif i == 1:
            t = x[0] + 1
            im1 = m
        elif 1 < i:
            t = x[i-1]
            im1 = i - 1

        x = x.at[i-1].set(0)
        x = x.at[im1-1].set(x[im1-1] + 1)
        x = x.at[m-1].set(x[m-1] + t - 1)

        return x



    #OBJECTIVE FUNCTIONS
    #coeff: coefficients of polynomial
    #target: y-values given in data
    #VM: terms of polynomial given by vandermonde_jax
    #alpha: ridge parameter
    def lst_sq(self, coeff, target, VM):
        res_sq = jnp.sum(jnp.square(target - VM@coeff))
        return res_sq
    
    def lst_sq_w(self, coeff, target, target_err, VM):
        w_res_sq = jnp.sum(jnp.square((target - VM@coeff)/target_err))
        return w_res_sq

    def ridge_obj_w(self, coeff, target, target_err, VM, alpha):          
        w_res_sq = jnp.sum(jnp.square((target - VM@coeff)/target_err))
        penalty = alpha*(coeff@coeff)
        return w_res_sq + penalty
    
    def ridge_obj(self, coeff, target, VM, alpha):
        res_sq = jnp.sum(jnp.square(target - VM@coeff))
        penalty = alpha*(coeff@coeff)
        return res_sq + penalty
    
    def lasso_obj_w(self, coeff, target, target_err, VM, alpha):          
        w_res_sq = jnp.sum(jnp.square((target - VM@coeff)/target_err))
        penalty = alpha*jnp.sum(jnp.abs(coeff))
        return w_res_sq + penalty
    
    def lasso_obj(self, coeff, target, VM, alpha):
        res_sq = jnp.sum(jnp.square(target - VM@coeff))
        penalty = alpha*jnp.sum(jnp.abs(coeff))
        return res_sq + penalty
