import h5py
from matplotlib import pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
from functools import partial


class Polyfit:
    def __init__(self, npz_file, **kwargs):
        """ 
        If 'input_h5' are 'order' are given, fit polynomials of order order to each bin in input_h5,
        
        If 'input_h5' and 'order' are not given, 
        The pceoff, chi2res dicts have the binids as keys with leading '\' characters stripped.

        Mandatory arguments:
        npz_file -- filepath for npz file storing results
        Optional Keyword Arguments:
        covariance -- whether or not to create covariance matrix. permanent for this object
        input_h5 -- the name of an h5 file with MC run results
        order -- the order of polynomial to fit each bin with
        """

        if len(kwargs) == 3 and 'input_h5'  and 'order' and 'covariance' in kwargs.keys():
            self.input_h5 = kwargs['input_h5']
            self.order = kwargs['order']

            f = h5py.File(self.input_h5, "r")
            self.dim = len(f['params'][0])
            self.num_coeffs = self.numCoeffsPoly(self.dim, self.order) 
            self.bin_ids = [x.decode() for x in f.get("index")[:]]
            # the index keys bin names to the array indexes in f.get(index) with binids matching that bin name
            self.index = {}
            # If bin not a key in index yet, start a new list as its value. Append count to bin's value. 
            [self.index.setdefault(bin.split('#')[0], []).append(count) for count,bin in enumerate(self.bin_ids)]
            # some bins_names have '[]' and actually represent uncertainties. We will have to handle this outside of the npz file.

            
            # keys observable names to their error bin names
            self.obs_index = {}
            [self.obs_index.setdefault(bin_name.split('[')[0], []).append(bin_name) if ('[') in bin_name 
                else self.obs_index.setdefault(bin_name.split('[')[0], []) for bin_name in self.index.keys()]
            print("Fitting observables: ", list(self.obs_index.keys()))

            self.p_coeffs, self.chi2ndf, self.res = [],[],[]
            self.cov = [] if kwargs['covariance'] else None 

            self.X = jnp.array(f['params'][:], dtype=jnp.float64) #num_mc_runs * dim array
            self.Y = jnp.array(f['values'][:], dtype=jnp.float64) #num_bins * num_mc_runs array
            VM = self.vandermonde_jax(self.X, self.order)
            #optimize this loop later
            #we do want to fit curves to every bin name (values and uncertainties) for w_err
            #TODO: make uncertainty symmetric if we can't do asymmetric
            for bin_id in self.bin_ids:
                bin_name, bin_number = bin_id.split('#')[0], int(bin_id.split('#')[1])
                bin_Y = jnp.array(f['values'][self.index[bin_name][int(bin_number)]])
                
                #polynomialapproximation.coeffsolve2 code
                bin_p_coeffs, bin_res, rank, s  = jnp.linalg.lstsq(VM, bin_Y, rcond=None)

                surrogate_Y = self.surrogate(self.X, bin_p_coeffs)
                bin_chi2 = jnp.sum(jnp.divide(jnp.power((bin_Y - surrogate_Y), 2), surrogate_Y))

                self.p_coeffs.append(bin_p_coeffs.tolist())
                self.res.append(bin_res[0]) #bin_res comes out of lstsq as a list
                self.chi2ndf.append(bin_chi2/(self.num_coeffs-1)) #because it's supposed to be /ndf
                
                def res_sq(coeff):
                    return jnp.sum(jnp.square(bin_Y-VM@coeff))
                def Hessian(func):
                    return jax.jacfwd(jax.jacrev(res_sq))
                #polynomialapproximation.fit code
                if kwargs['covariance']:
                    #cov = np.linalg.inv(VM.T@VM)
                    cov = jnp.linalg.inv(Hessian(res_sq)(bin_p_coeffs))
                    fac = bin_res / (VM.shape[0]-VM.shape[1])
                    bin_idn = self.bin_idn(bin_id)
                    self.cov.append(cov*fac)
                    #print(bin_id, "\n", bin_p_coeffs, "\n", jnp.sqrt(jnp.diagonal(self.cov[bin_idn])), "\nend")
                    #print(bin_id, bin_p_coeffs, self.cov[bin_id], " end")
                self.save(npz_file)

        elif len(kwargs) == 1 and 'covariance' in kwargs.keys():
            self.p_coeffs, self.res, self.chi2ndf, self.bin_ids, self.X, self.Y\
                = np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0)
            self.cov = [] if kwargs['covariance'] else None 
            self.order, self.dim = None, None
            self.merge(npz_file, True)
        else:
            print('invalid args given to polyfit')

    def merge(self, all_npz, new = False):
        """ Merge new data to existing class variables.
        For now, if new data has same bin ids as old data, both copies are kept.
        """
        all_dict = jnp.load(all_npz, allow_pickle=True)
        if self.order is None and self.dim is None:
            self.order, self.dim = all_dict['order'], all_dict['dim']
        elif self.order != all_dict['order'] or self.dim != all_dict['dim']:
            print("merging data with different order/dim is not a pro gamer move(error)")
        self.num_coeffs = self.numCoeffsPoly(self.dim, self.order) 
        def join_new(str): #jnp->numbers only
            setattr(self, str, jnp.array(all_dict[str]) if new else jnp.concatenate([attr, all_dict[str]]))
        for str in ['p_coeffs', 'chi2ndf', 'res', 'X', 'Y']: join_new(str)
        if not self.cov is None: join_new('cov')
        # We recalculate index, obs_index from bin_ids. I decided to just store bin_ids because 
        # 1. npz likes np lists only and
        # 2. when we merge, iterating through index is inevitable``, we can't just concatenate.
        self.bin_ids = all_dict['bin_ids'] if new else np.concatenate([self.bin_ids, all_dict['bin_ids']])
        self.index = {}
        [self.index.setdefault(bin.split('#')[0], []).append(count) for count,bin in enumerate(self.bin_ids)]
        self.obs_index = {}
        [self.obs_index.setdefault(bin_name.split('[')[0], []).append(bin_name) 
            for bin_name in self.index.keys() if ('[') in bin_name]


    def save(self, all_npz):
        """ Save data to given fileplaths

        Mandatory arguments:
        p_coeffs_npz -- filepath for npz file of the polynomial coefficients
        chi2res_npz -- filepath for npz file of the chi2.ndf and residuals
        Optional Keyword Arguments:
        cov_npz -- filepath for npz of the covariance matrix
        """
        all_dict = {}
        all_dict['p_coeffs'] = self.p_coeffs
        all_dict['chi2ndf'] = self.chi2ndf
        all_dict['res'] = self.res
        all_dict['cov'] = self.cov 
        all_dict['bin_ids'] = self.bin_ids
        all_dict['X'] = self.X
        all_dict['Y'] = self.Y
        all_dict['dim'] = self.dim
        all_dict['order'] = self.order
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
        plt.stairs([y for y in self.Y[self.bin_idn(bin_id), :]], edges, label = 'Target Data')
        surrogate, chi2, res, cov = self.get_surrogate_func(bin_id)
        surrogate_y = surrogate(self.X)
        plt.stairs(surrogate_y, edges, label = 'Surrogate(Tuned Parameters)')
        plt.legend()

    def get_surrogate_func(self, bin_id):
        """
        Takes bin_id, either in format bin_name#bin_number, or int: place of the bin in pceoff etc.
        Returns a surrogate that does not need pceoffs as an input; just takes x(param values).
        Returns chi2/ndf of fit, residual(s), and cov if it exists.
        """
        bin_idn = self.bin_idn(bin_id) if type(bin_id) is str else bin_id        
        return partial(self.surrogate, p_coeffs = self.p_coeffs[bin_idn]), self.chi2ndf[bin_idn], self.res[bin_idn], \
            self.cov[bin_idn] if self.cov else None
    
    def surrogate(self, x, p_coeffs):
        """
        Takes a list/array of param sets, and returns the surrogate's estimate for their nominal values.
        The length of the highest dimension of the list/array is the number of params.
        """
        x = np.array(x)
        dim = x.shape[-1]
        #has to be mutable the way this is written
        term_powers = np.zeros(dim)
        pvalues = []
        last_axis = None if len(x.shape) == 1 else len(x.shape)-1
        for pcoeff in p_coeffs:
            #appending the value of the term for each param set given
            pvalues.append(pcoeff*np.prod(x**term_powers, axis = last_axis))
            term_powers = self.mono_next_grlex(dim, term_powers)
        #add along first axis to sum the terms for each param set
        return np.sum(pvalues, axis=None if len(x.shape) == 1 else 0)
    
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

        #We will take params to the power of grlex_pow element-wise
        if dim == 1:
            grlex_pow = jnp.array(range(order+1))
        else:
            term_list = [[0]*dim]
            for i in range(1, self.numCoeffsPoly(dim, order)):
                term_list.append(self.mono_next_grlex(dim, term_list[-1][:]))
            grlex_pow = jnp.array(term_list)
        
        if dim == 1:
            V = jnp.zeros((len(params), self.numCoeffsPoly(dim, order)), dtype=jnp.float64)
            for a, p in enumerate(params): 
                V = V.at[a].set(p**grlex_pow)
            return V
        else:
            V = jnp.power(params, grlex_pow[:, jnp.newaxis])
            return jnp.prod(V, axis=2).T
    def bin_idn(self, bin_id):
        # Convert bin_id in fomart bin_name#bin_number to binidn
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
            x[m-1] = 1
            return x
        elif i == 1:
            t = x[0] + 1
            im1 = m
        elif 1 < i:
            t = x[i-1]
            im1 = i - 1

        x[i-1] = 0
        x[im1-1] = x[im1-1] + 1
        x[m-1] = x[m-1] + t - 1

        return x


