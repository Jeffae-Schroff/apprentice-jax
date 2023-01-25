import h5py
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
from jax.config import config
import scipy.optimize as opt
import numpy as np
config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')
from functools import partial


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


    def __init__(self, npz_file, **kwargs):
        """ 
        If 'input_h5' are 'order' are given, fit polynomials of order order to each bin in input_h5,
        
        If 'input_h5' and 'order' are not given, 
        The pceoff, chi2res dicts have the binids as keys with leading '\' characters stripped.

        Mandatory arguments:
        npz_file -- filepath for npz file storing results
        Keyword Arguments:
        covariance -- whether or not to create covariance matrix. permanent for this object
        Optional Keyword Arguments (use to create fits):
        input_h5 -- the name of an h5 file with MC run results
        order -- the order of polynomial to fit each bin with
        reg_mode -- regression mode, default to lst_sq
        """

        if 'input_h5' in kwargs.keys() and 'order' in kwargs.keys() and 'covariance' in kwargs.keys():
            self.input_h5 = kwargs['input_h5']
            self.order = kwargs['order']
            self.has_cov = kwargs['covariance']
            self.reg_mode = 'lst_sq'
            self.reg_param = 0
            if 'reg_mode' in kwargs.keys():
                self.reg_mode = kwargs['reg_mode']
                self.reg_param = kwargs['reg_param']

            f = h5py.File(self.input_h5, "r")
            self.X = jnp.array(f['params'][:], dtype=jnp.float64) #num_mc_runs * dim array
            self.Y = jnp.array(f['values'][:], dtype=jnp.float64) #num_bins * num_mc_runs array
            self.bin_ids = np.array([x.decode() for x in f.get("index")[:]])
            #filter out bins with invalid value for any mc_run
            invalid = jnp.array(jnp.any(abs(self.Y[:,:]) > 1000, axis = 1).nonzero()[0])
            print("Filtering", len(invalid), "bins for invalid input")
            self.Y = jnp.delete(self.Y, invalid, axis=0)
            self.bin_ids = np.delete(self.bin_ids, invalid)
            
            # the index keys bin names to the array indexes in f.get(index) with binids matching that bin name
            self.index = {}
            # If bin not a key in index yet, start a new list as its value. Append count to bin's value. 
            [self.index.setdefault(bin.split('#')[0], []).append(count) for count,bin in enumerate(self.bin_ids)]            
            # keys observable names to their error bin names
            self.obs_index = {}
            [self.obs_index.setdefault(bin_name.split('[')[0], []).append(bin_name) if ('[') in bin_name 
                else self.obs_index.setdefault(bin_name, []) for bin_name in self.index.keys()]
            
            self.dim = self.X.shape[1]
            self.num_coeffs = self.numCoeffsPoly(self.dim, self.order) 
            VM = self.vandermonde_jax(self.X, self.order)
            #optimize this loop later
            #we do want to fit curves to every bin name (values and uncertainties) for w_err
            #TODO: make uncertainty symmetric if we can't do asymmetric

            self.p_coeffs, self.chi2ndf, self.res = [],[],[]
            if self.has_cov: self.cov = []
            debug=1 ##TODO: DELETE
            for bin_count, bin_id in enumerate(self.bin_ids):
                if 'num_bins' in kwargs.keys() and bin_count > kwargs['num_bins']:
                    break
                print("\rFitting {:d} of {:d}: {:60s}".format(bin_count, self.Y.shape[0], bin_id), end='')
                
                bin_Y = self.Y[self.bin_idn(bin_id),:]
                
                #polynomialapproximation.coeffsolve2 code
                obj_args = (bin_Y, VM, self.reg_param)
                if self.reg_mode == 'ridge':
                    guess = jnp.zeros((VM.shape[1],), dtype=jnp.float32)
                    c_opt = opt.minimize(self.ridge_obj, guess, args=obj_args, method='Nelder-Mead')
                    bin_p_coeffs = c_opt.x
                    bin_res = [jnp.sum(jnp.square(bin_Y-VM@bin_p_coeffs))]
                else:
                    bin_p_coeffs, bin_res, rank, s  = jnp.linalg.lstsq(VM, bin_Y, rcond=None)

                surrogate_Y = self.surrogate(self.X, bin_p_coeffs)
                bin_chi2 = jnp.sum(jnp.divide(jnp.power((bin_Y - surrogate_Y), 2), surrogate_Y))

                self.p_coeffs.append(bin_p_coeffs.tolist())
                self.res.append(bin_res[0]) #bin_res comes out of lstsq as a list
                self.chi2ndf.append(bin_chi2/(self.num_coeffs-1)) #because it's supposed to be /ndf
                
                #Calculating covariance of coefficients using inverse Hessian
                def res_sq(coeff):
                    return jnp.sum(jnp.square(bin_Y-VM@coeff))
                def ridge(coeff):
                    return self.ridge_obj(coeff, bin_Y, VM, self.reg_param)
                def Hessian(func):
                    return jax.jacfwd(jax.jacrev(func))
                #polynomialapproximation.fit code
                if self.has_cov:
                    if self.reg_mode == 'ridge':
                        pcov = jnp.linalg.inv(Hessian(ridge)(bin_p_coeffs))
                    else:
                        pcov = jnp.linalg.inv(Hessian(res_sq)(bin_p_coeffs))
                    fac = bin_res[0] / (VM.shape[0]-VM.shape[1])
                    self.cov.append(pcov*fac)

                    # #Old code!
                    """cov = np.linalg.inv(VM.T@VM)
                    fac = bin_res / (VM.shape[0]-VM.shape[1])"""
                    if debug==1:
                        print("\nScaling factor in polyfit: ", fac)
                        
    
                        with jnp.printoptions(precision=3, linewidth=1000, suppress=True, floatmode="fixed"):
                            print("Y: ", bin_Y)
                            print("bin_p_coeff: ", bin_p_coeffs)
                            print("\nVM matrix: \n", VM)
                            print("\nInverse Hessian of lst_sq: \n ", jnp.linalg.inv(Hessian(res_sq)(bin_p_coeffs)))
                            print("\nInverse Hessian of ridge: \n ", jnp.linalg.inv(Hessian(ridge)(bin_p_coeffs)))
                            jnp.save('polyfit_inv_hess_lst_sq', jnp.linalg.inv(Hessian(res_sq)(bin_p_coeffs)))
                        debug=0
                    # self.cov.append(cov*fac)

                    #print(bin_id, "\n", bin_p_coeffs, "\n", jnp.sqrt(jnp.diagonal(self.cov[bin_idn])), "\nend")
                    #print(bin_id, bin_p_coeffs, self.cov[bin_id], " end")
            self.save(npz_file)

        #Initialize with all attributes empty
        elif 'covariance' in kwargs.keys():
            self.has_cov = kwargs['covariance']
            self.merge(npz_file, new = True)
        else:
            print('invalid args given to polyfit')

    def merge(self, all_npz, new = False):
        """ Merge new data to existing class variables.
        For now, if new data has same bin ids as old data, both copies are kept.
        """
        all_dict = jnp.load(all_npz, allow_pickle=True)
        if new:
            self.order, self.dim = all_dict['order'], all_dict['dim']
        elif self.order != all_dict['order'] or self.dim != all_dict['dim']:
            print("merging data with different order/dim is not allowed(error)")
        self.num_coeffs = self.numCoeffsPoly(self.dim, self.order)

        jnp_vars = ['p_coeffs', 'chi2ndf', 'res', 'X', 'Y']
        if self.has_cov: jnp_vars.append('cov')
        for str in jnp_vars: #jnp: numbers only
            if new:
                setattr(self, str, jnp.array(all_dict[str]))
            else:
                setattr(self, str, jnp.concatenate([getattr(self, str), all_dict[str]]))
        
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
        all_vars = ['p_coeffs', 'chi2ndf', 'res', 'X', 'Y', 'bin_ids', 'dim', 'order']
        if self.has_cov: all_vars.append('cov') 
        for str in all_vars:
            all_dict[str] = getattr(self, str)
        jnp.savez(all_npz, **all_dict)

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
        plt.stairs([y for y in self.Y[self.fits.bin_idn(bin_id), :]], edges, label = 'Target Data')
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
        return self.index[bin_id.split('#')[0]][int(bin_id.split('#')[1])]
    
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



    #NEW OBJECTIVE FUNCTIONS: RIDGE/LASSO
    #coeff: coefficients of polynomial
    #target: y-values given in data
    #VM: terms of polynomial given by vandermonde_jax
    #alpha: ridge parameter
    def ridge_obj(self, coeff, target, VM, alpha):
        res_sq = jnp.sum(jnp.square(target - VM@coeff))
        penalty = alpha*(coeff@coeff)
        return res_sq + penalty