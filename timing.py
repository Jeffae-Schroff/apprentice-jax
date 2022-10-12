import jax.numpy as jnp

def timing_van_jax(params, order):
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
        for i in range(1, numCoeffsPoly(dim, order)):
            term_list.append(mono_next_grlex(dim, term_list[-1][:]))
        grlex_pow = jnp.array(term_list)
    
    if dim == 1:
        V = jnp.zeros((len(params), numCoeffsPoly(dim, order)), dtype=jnp.float64)
        for a, p in enumerate(params): 
            V = V.at[a].set(p**grlex_pow)
        return V
    else:
        V = jnp.power(params, grlex_pow[:, jnp.newaxis])
        return jnp.prod(V, axis=2).T

def numCoeffsPoly(dim, order):
    """
    Number of coefficients a dim-dimensional polynomial of order order has (C(dim+order, dim)).
    """
    ntok = 1
    r = min(order, dim)
    for i in range(r):
        ntok = ntok * (dim + order - i) / (i + 1)
    return int(ntok)

#Next term of grlex ordering
def mono_next_grlex(m, x):
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

def van_jax_comp(params, order):
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
        for i in range(1, numCoeffsPoly(dim, order)):
            term_list.append(mono_next_grlex(dim, term_list[-1][:]))
        grlex_pow = jnp.array(term_list)
    return grlex_pow




def poly_gen_C(X, n):
    V = jnp.power(X, comp_gen_C(X.size, n)[:, jnp.newaxis])
    return jnp.prod(V, axis=2).T

def comp_gen_C(m, n):
    one_count = jnp.arange(0, n + 1)
    parts = jnp.stack(jnp.meshgrid(*jnp.tile(one_count, (m,1)))).T.reshape(-1,m)
    parts = parts[jnp.sum(parts, axis=1)<n+1]
    return parts





def poly_gen_D(X, n):
    comp = jnp.array(comp_gen_D(X.size, n))
    V = jnp.power(X, comp[:, jnp.newaxis])
    return jnp.prod(V, axis=2).T

#Generate all compositions <=n of length k
def comp_gen_D(k, n):
    return tuple(comp_D(k, n))

def comp_D(k, n):
    def out_loop(n, k, t):
        allowed = range(0, n + 1, 1)
        def in_loop(n, k, t):
            if k == 0:
                if n == 0:
                    yield t
            elif k == 1:
                if 0 <= n:
                    yield t + (n,)
            elif 0 <= n:
                for v in allowed:
                    yield from in_loop(n - v, k - 1, t + (v,))
        m = 0
        while m <= n:
            yield from in_loop(m, k, t)
            m += 1
    return out_loop(n, k, ())
