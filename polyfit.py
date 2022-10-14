import h5py
import numpy as np
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
from functools import partial
import json

class Polyfit:
    def __init__(self, pcoeffs_npz, chi2res_npz, **kwargs):
        """ 
        If 'input_h5' are 'order' are given, fit polynomials of order order to each bin in input_h5,
        then store them in npz files(pcoeffs_npz, chi2res_npz, cov_npz(if given))
        If 'input_h5' and 'order' are not given, it assumes pcoeffs_npz, chi2res_npz
        store fits from a previous polyfit instance
        The pceoff, chi2res dicts have the binids as keys with leading '\' characters stripped.

        Mandatory arguments:
        pcoeffs_npz -- filepath for npz file of the polynomial coefficients
        chi2res_npz -- filepath for npz file of chi2.ndf and residuals
        Optional Keyword Arguments:
        cov_npz -- filepath for npz file of covariance matrix
        input_h5 -- the name of an h5 file with MC run results
        order -- the order of polynomial to fit each bin with
        """

        if 'input_h5' in kwargs.keys() and 'order' in kwargs.keys():
            self.input_h5 = kwargs['input_h5']
            self.order = kwargs['order']

            f = h5py.File(self.input_h5, "r")

            # key bin names to the array indexes in f.get(index) with binids matching that bin name
            self.index = {}
            bin_ids = [x.decode() for x in f.get("index")[:]]
            # If key not in index yet, start a new list as its value. Append i to key's value. Strip '/' for valid filenames
            [self.index.setdefault(bin.replace('/', '').split('#')[0], []).append(i) for i,bin in enumerate(bin_ids)]
            self.dim = len(f['params'][0])

            self.pcoeffs = {}
            self.res = {}
            self.chi2 = {}
            if 'cov_npz' in kwargs.keys(): self.cov = {}
            for bin_id in bin_ids:
                bin_id = bin_id.replace('/', '')
                bin_name, bin_number = bin_id.split('#')[0], int(bin_id.split('#')[1])
                X = np.array(f['params'][:], dtype=np.float64)
                Y = np.array(f['values'][self.index[bin_name][int(bin_number)]])
                VM = self.vandermonde_jax(X, self.order)

                #polynomialapproximation.coeffsolve2 code
                bin_pcoeffs, bin_res, rank, s  = np.linalg.lstsq(VM, Y, rcond=None)

                surrogate_Y = self.surrogate(X, bin_pcoeffs)
                bin_chi2 = np.sum(np.divide(np.power((Y - surrogate_Y), 2), surrogate_Y))

                self.pcoeffs[bin_id] = bin_pcoeffs.tolist()
                self.res[bin_id] = bin_res
                self.chi2[bin_id] = bin_chi2/self.numCoeffsPoly(self.dim, self.order) #because it's supposed to be /ndf
                
                #polynomialapproximation.fit code
                if 'cov_npz' in kwargs.keys():
                    cov = np.linalg.inv(VM.T@VM)
                    fac = bin_res / (VM.shape[0]-VM.shape[1])
                    self.cov[bin_id] = cov*fac
            if 'cov_npz' in kwargs.keys(): 
                self.save(pcoeffs_npz, chi2res_npz, cov_npz = kwargs['cov_npz'])
            else: 
                self.save(pcoeffs_npz, chi2res_npz)

        elif len(kwargs) == 0 or ('cov_npz' in kwargs.keys() and len(kwargs) == 1):
            self.pcoeffs, self.res, self.chi2 = {}, {}, {}
            if 'cov_npz' in kwargs.keys(): 
                self.cov = {}
                self.merge(pcoeffs_npz, chi2res_npz, cov_npz = kwargs['cov_npz'])
            else: 
                self.merge(pcoeffs_npz, chi2res_npz)
        else:
            print('invalid args given to polyfit')

    def merge(self, pcoeffs_npz, chi2res_npz, **kwargs):
        """ Merge new data to existing class variables.
        If new data has any bins with the same keys as old data's bins, it will replace the old data.

        Mandatory arguments:
        pcoeffs_npz -- filepath for npz file of the polynomial coefficients
        chi2res_npz -- filepath for npz file of the chi2.ndf and residuals
        Optional Keyword Arguments:
        cov_npz -- filepath for npz of the covariance matrix
        """
        self.pcoeffs = {**self.pcoeffs, **np.load(pcoeffs_npz, allow_pickle=True)}
        chi2res = np.load(chi2res_npz, allow_pickle=True)
        self.chi2 = {**self.chi2, **{b: chi2res[b][0] for b in chi2res.keys()}}
        self.res = {**self.res, **{b: chi2res[b][1] for b in chi2res.keys()}}
        if 'cov_npz' in kwargs.keys():
            self.cov = {**self.cov, **np.load(kwargs['cov_npz'], allow_pickle=True)}

    def save(self, pcoeffs_npz, chi2res_npz, **kwargs):
        """ Save data to given fileplaths

        Mandatory arguments:
        pcoeffs_npz -- filepath for npz file of the polynomial coefficients
        chi2res_npz -- filepath for npz file of the chi2.ndf and residuals
        Optional Keyword Arguments:
        cov_npz -- filepath for npz of the covariance matrix
        """
        np.savez(pcoeffs_npz, **self.pcoeffs)
        chi2res = {b: np.array([self.chi2[b], self.res[b]]) for b in self.chi2.keys()}
        np.savez(chi2res_npz, **chi2res)
        if 'cov_npz' in kwargs.keys(): np.savez(kwargs['cov_npz'], **self.cov)

    def get_XY(self, bin_id):
        # probably temp for testing
        f = h5py.File(self.input_h5, "r")
        bin_id = bin_id.replace('/', '')
        bin_name, bin_number = bin_id.split('#')[0], int(bin_id.split('#')[1])
        return np.array(f['params'][:], dtype=np.float64),np.array(f['values'][self.index[bin_name][int(bin_number)]])

    def get_surrogate_func(self, bin_id):
        """
        Takes bin_id, either in format bin_name#bin_number, or int: place of the bin in dict (not recommended)
        Returns a surrogate that does not need pceoffs as an input; just takes x(param values).
        Returns chi2/ndf of fit, residual(s), and cov if it exists.
        """
        if type(bin_id) is int:
            bin_id = list(self.pcoeffs.keys())[bin_id]
        else:
            bin_id = bin_id.replace('/', '')
        return partial(self.surrogate, pcoeffs = self.pcoeffs[bin_id]), self.chi2[bin_id], self.res[bin_id], \
            self.cov[bin_id] if hasattr(self, 'cov') else None
    
    def surrogate(self, x, pcoeffs):
        """
        Takes a list/array of param sets, and returns the surrogate's estimate for their nominal values.
        The length of the highest dimension of the list/array is the number of params.
        """
        x = np.array(x)
        dim = x.shape[-1]
        term_powers = dim*[0]
        pvalues = []
        last_axis = None if len(x.shape) == 1 else len(x.shape)-1
        for pcoeff in pcoeffs:
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


