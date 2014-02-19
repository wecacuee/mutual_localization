import numpy as np
import sympy
assert float(sympy.__version__[:3]) >= 0.7
from sympy.matrices import Matrix
import sympy.mpmath.calculus.optimization as mpopt
import itertools as it
import utils
from memoize import memoized
from sympy.utilities.lambdify import lambdify

import log
import logging
logger = log.getLogger(__name__)
logger.setLevel(logging.WARN)

TOL = 1e-6

def _sorted_combinations(iter, n):
    """
    >>> list(_sorted_combinations([1, 2, 3, 4], 2))
    [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    """
    return it.combinations(iter,n)

def _polynomial_unknowns(x):
    return ([i**2 for i in x]
            + list(map(lambda c: c[0]*c[1], _sorted_combinations(x, 2)))
            + list(x) + [1])

_sym_count = 0
def _next_sym():
    global _sym_count
    sym =  sympy.Symbol('c%d' % _sym_count)
    _sym_count += 1
    return sym

def _gensyms(n):
    return sympy.symbols(' '.join('x%d' % i for i in range(n)))

def _sym_vector(name):
    return Matrix(
        sympy.symbols(['%s%d' % (name, i) for i in range(3)],
                      real=True))

def _map_to_coeffs(dict_, value):
    sym = _next_sym()
    dict_[sym] = value
    return sym

def _generate_sym_polynomials(nf1, nf2):
    frame1scaled = [(_sym_vector('p%d' % i), _sym_vector('q%d' % i))
                    for i in range(nf1)]
    frame2scaled = [(_sym_vector('p%d' % i), _sym_vector('q%d' % i))
                    for i in range(nf1, nf1 + nf2)]
    n = nf1 + nf2
    nmonomials = (n + 2)*(n+1)/ 2
    neqns = n * (n-1)/ 2
    P = sympy.zeros((neqns, nmonomials))
    nrow = 0
    coeffs = dict()
    for ipoint, jpoint in _sorted_combinations(enumerate(frame1scaled), 2):
        i, (pi, qi) = ipoint
        j, (pj, qj) = jpoint
        P[nrow, i] = 1
        P[nrow, j] = 1
        P[nrow, n-1 + i*(2*n  - i-1)/2 + j - i] = \
                _map_to_coeffs(coeffs, -2 * pi.dot(pj))
        P[nrow, nmonomials - 1] = _map_to_coeffs(coeffs, 
                                                 - (qi - qj).dot(qi - qj))
        nrow += 1

    f2start = len(frame1scaled)
    for ipoint, jpoint in it.product(enumerate(frame1scaled),
                                     enumerate(frame2scaled, f2start)):
        i, (pi, qi) = ipoint
        j, (pj, qj) = jpoint
        P[nrow, i] = 1
        P[nrow, j] = -1
        P[nrow, nmonomials -n-1 + i] = _map_to_coeffs(coeffs, -2 * pi.dot(pj))
        P[nrow, nmonomials -n-1 + j] = _map_to_coeffs(coeffs, 2 * qi.dot(qj))
        P[nrow, nmonomials - 1] = _map_to_coeffs(coeffs, 
                                                 pj.dot(pj) - qi.dot(qi))
        nrow += 1
    for ipoint, jpoint in _sorted_combinations(enumerate(frame2scaled, f2start), 2):
        i, (pi, qi) = ipoint
        j, (pj, qj) = jpoint
        P[nrow, i] = 1
        P[nrow, j] = 1
        P[nrow, n-1 + i*(2*n  - i-1)/2 + j - i] = \
                _map_to_coeffs(coeffs, -2 * qi.dot( qj))
        P[nrow, nmonomials - 1] = _map_to_coeffs(coeffs, 
                                                 - (pi - pj).dot( pi - pj))

    return P, coeffs

@memoized
def generate_polynomials(nf1, nf2):
    n = nf1 + nf2
    P, coeffs = _generate_sym_polynomials(nf1, nf2)
    xsyms = _gensyms(n)
    x = Matrix(xsyms)
    X = Matrix(_polynomial_unknowns(x))
    return P * X, xsyms, coeffs

def get_coeffs(expr, x):
    poly = sympy.Poly(expr, x)
    n = poly.degree()
    coef = poly.coeffs()
    assert len(coef) == n + 1
    return coef

def sylvester_det(p4, p2):
    p4degree = len(p4) - 1
    p2degree = len(p2) - 1
    smatsize = p4degree + p2degree
    nzeros4 = (smatsize - len(p4))
    nzeroes2 = (smatsize - len(p2))
    smatlist = []
    for i in range(nzeros4 + 1):
        smatlist.append([0]*i + p4 + [0]*(nzeros4 - i))
    for i in range(nzeroes2 + 1):
        smatlist.append([0]*i + p2 + [0]*(nzeroes2 - i))

    smat = Matrix(smatlist)
    det = smat.det(method='berkowitz')
    det = sympy.expand(det)
    return det

def poly_free_symbols(add_eq):
    add_eq = add_eq.as_poly()
    return getattr(add_eq, 'free_symbols', # version 0.7.1rc1
            set(getattr(add_eq, 'symbols', []))) #version 0.6
    
def _polyidx_by_symbols(eqns, xsyms):
    sym2poly = dict()
    sympair2poly = dict()
    for i, eq in enumerate(eqns):
        x, y = sorted(poly_free_symbols(eq) & set(xsyms))
        sympair2poly[(x, y)] = i
        sym2poly.setdefault(x, []).append(i)
        sym2poly.setdefault(y, []).append(i)
    return sym2poly, sympair2poly

def _eliminate_one(p4d, quad, x):
    """
    eliminates x from polynomial p4d and quadratic quad
    """
    p4coeff = get_coeffs(p4d, x)
    quadcoeff = get_coeffs(quad, x)
    return sylvester_det(p4coeff, quadcoeff)

def eliminate(eqns, xsyms):
    sym2polyidx, sympair2polyidx = _polyidx_by_symbols(eqns, xsyms)
    return _eliminate(eqns, xsyms, sym2polyidx, sympair2polyidx)

def _eliminate(eqns, xsyms, sym2polyidx, sympair2polyidx):
    n = len(xsyms)
    nfinals = int((n - 1) * (n-2) / 2)
    final_eqns = []
    ispolyused = dict()
    for sym in xsyms[:-1]:
        polyidx = sym2polyidx[sym]
        newpoly = [_eliminate_one(eqns[polyidx[0]], eqns[i], sym)
                   for i in polyidx[1:] if not ispolyused.get(i)]
        for np in newpoly:
            npsyms = poly_free_symbols(np) & set(xsyms)
            # Here we depend on the knowledge that our systems of equations
            # are always bi-variate.
            x, y = sorted(npsyms)
            p2idx = sympair2polyidx[(x, y)]
            feq = _eliminate_one(np, eqns[p2idx], x)
            final_eqns.append((y, feq.as_poly(y).coeffs()))
            ispolyused[p2idx] = True
    return final_eqns

def ordered_pi_qi(nf1, nf2):
    pi_qi = list()
    for i in range(nf1 + nf2):
        pi_qi.extend(_sym_vector('p%d'%i))
        pi_qi.extend(_sym_vector('q%d'%i))
    return pi_qi

def _eval_coeffs_lambdified(coeff_lbd_sorted, frame1scaled, frame2scaled):
    piqi_values_sorted = []
    # Depends on the order in which list is constructed 
    # change ordered_pi_qi() if changing the order
    for (pi, qi) in (frame1scaled + frame2scaled):
        piqi_values_sorted.extend(pi)
        piqi_values_sorted.extend(qi)

    evaluated = dict()
    for k, v in coeff_lbd_sorted:
        evaluated[k] = v(*piqi_values_sorted)
    return evaluated

def lambdify_final_eqns(coeff_keys_sorted, final_eqns):
    lambdified_final_eqns = []
    for freev, coeff_expr in final_eqns:
        leq = [lambdify(coeff_keys_sorted, expr) for expr in coeff_expr]
        lambdified_final_eqns.append((freev, leq))
    return lambdified_final_eqns

@memoized
def _symbolic_computations(nf1, nf2):
    """
    Params
    nf1: number of scaled points in frame 1
    nf2: number of scaled points in frame 2

    Returns:
        P     :
            System of multivariate polynomials
        xsyms :
            Unknown variables in system of equations P
        coeff_sorted :
            A list of pair of real coefficients in P and functions in terms of
            ordered_pi_qi. Use function _eval_coeffs_lambdified() to evaluate coeff
        sym2polyidx :
            Eqns in P grouped by symbols
        sympair2polyidx :
            Eqns in P grouped by symbol pairs assuming each equation in P
            contains only two variables.
        lambda_final_eqns :
            Resultant univariate polynomial equations as a list of free_symbol
            and lambda_expression pair. The lambda expression takes arguments
            in the same order as coeff_sorted keys.
    """
    assert nf1 >= 1 and nf1 <= 2
    assert nf2 >= 1 and nf2 <= 2
    P, xsyms, coeff = generate_polynomials(nf1, nf2)
    logger.debug("Equation system: {0}".format(P))
    logger.debug("\twhere : {0}".format(sorted(coeff.items(), key=lambda x:x[0])))
    sym2polyidx, sympair2polyidx = _polyidx_by_symbols(P, xsyms)
    final_eqns = _eliminate(P, xsyms, sym2polyidx, sympair2polyidx)

    # Convert sympy expressions to lambda functions
    coeff_keys_sorted = sorted(coeff.keys())
    piqi_order = ordered_pi_qi(nf1, nf2)
    coeff_expr = [lambdify(piqi_order, coeff[k]) for k in coeff_keys_sorted]
    coeff_sorted = zip(coeff_keys_sorted, coeff_expr)
    lambdified_final_eqns = lambdify_final_eqns(coeff_keys_sorted, final_eqns)

    return (P, xsyms, coeff_sorted, sym2polyidx, sympair2polyidx,
            lambdified_final_eqns)

def numeric_find_scale_factors(frame1scaled, frame2scaled, tol=TOL):
    nf1 = len(frame1scaled)
    nf2 = len(frame2scaled)
    P, xsyms, coeff = generate_polynomials(nf1, nf2)
    coeff_evaluated = _eval_coeffs(coeff, frame1scaled, frame2scaled)
    Pnum = P.subs(coeff_evaluated)
    root_mat = sympy.solvers.nsolve(Pnum.T.tolist()[0], xsyms, [1., 1., 1., 1.],
                                solver=mpopt.MDNewton, tol=1e-2, verify=True,
                                   verbose=True)
    logger.debug("Got roots {0}".format(root_mat))
    return [np.array([float(s) for s in scales]) for scales in
                       root_mat.T.tolist()]

def scale_factors_from_quat(quat, trans, frame1scaled, frame2scaled):
    T = utils.transform_from_quat(quat, trans)
    return scale_factors_from_result(T, frame1scaled, frame2scaled)

def scale_factors_from_result(T, frame1scaled, frame2scaled):
    Tinv = utils.transform_inv(T)
    scale_factors = [utils.apply_transform(Tinv, f2)/f1
                     for f1, f2 in frame1scaled]
    scale_factors += [utils.apply_transform(T, f1)/f2
                      for f1, f2 in frame2scaled]
    return scale_factors

def add_to_poly(eq, coeff_evaluated):
    return eq.subs(coeff_evaluated).as_poly()

def eqn_as_sym_n_coeffs(lbd_final_eqns, coeff_eval_sorted):
    #numeric_poly = [add_to_poly(eq, coeff_evaluated) for eq in final_eqns]
    # Depending on the knowledge that the final polynomials are univariate. We
    # get dictionary from sym to the polynomial (polynomial is now represented
    # as an list of coefficients)
    #symsncoeffs = [(list(eq.free_symbols)[0], eq.coeffs())
    #               for eq in numeric_poly]
    symsncoeffs  = [(freev, [lmbda(*coeff_eval_sorted) for lmbda in lbd_exprs])
                    for freev, lbd_exprs in lbd_final_eqns]
    return symsncoeffs

def subs_poly_eqns(P, coeff_evaluated):
    Pnum = [eq.subs(coeff_evaluated) for eq in P]
    return Pnum

def evaluate_coefficients_in_equations(coeff_sorted, lbd_final_eqns, P, frame1scaled,
                                       frame2scaled):
    coeff_evaluated = _eval_coeffs_lambdified(coeff_sorted, frame1scaled, frame2scaled)
    Pnum = subs_poly_eqns(P, coeff_evaluated)
    symsncoeffs = eqn_as_sym_n_coeffs(lbd_final_eqns, 
                                     [coeff_evaluated[k]
                                      for k, v in coeff_sorted])
    return coeff_evaluated, Pnum, symsncoeffs

def get_real_roots(coeffs):
    return [sympy.re(r) for r in np.roots(coeffs) if _constraints(r)]

def min_scale_factors_projection_criteria(frame1scaled, frame2scaled):
    min_scalefactors = [np.min(
        [np.dot(mL_1, mR_1) / np.linalg.norm(mR_1) 
         for mR_1, mR_2 in frame1scaled])
        for mL_1, mL_2 in frame2scaled]
    min_scalefactors += [np.min(
        [np.dot(mR_2, mL_2) / np.linalg.norm(mL_2) 
        for mL_1, mL_2 in frame2scaled])
         for mR_1, mR_2 in frame1scaled]
    return min_scalefactors

def filter_scale_factors_by_projection_criteria(possible_scale_factors,
                                                frame1scaled,
                                                frame2scaled):
    min_scalefactors = min_scale_factors_projection_criteria(frame1scaled,
                                                frame2scaled)
    return [sf for sf in possible_scale_factors 
            if np.all(sf >= min_scalefactors)] 

def find_scale_factors(frame1scaled, frame2scaled, tol=TOL):
    nf1 = len(frame1scaled)
    nf2 = len(frame2scaled)
    P, xsyms, coeff_sorted, sym2polyidx, sympair2polyidx, lbd_final_eqns = \
            _symbolic_computations(nf1, nf2)

    coeff_evaluated, Pnum, symsncoeffs = evaluate_coefficients_in_equations(
        coeff_sorted, lbd_final_eqns, P, frame1scaled, frame2scaled)


    all_common_roots = list()
    for sym, coeffs in symsncoeffs:
        # consider only positive real roots
        roots = get_real_roots(coeffs)
        # for every variable we have atleast one equation that pairs it with
        # all the other variables. we have that correspondence available as
        # sym2polyidx
        common_roots = _find_common_roots(Pnum, roots, sym, sym2polyidx,
                                          sympair2polyidx)
        all_common_roots.extend(common_roots)

    if not len(all_common_roots):
        return []
    error_filtered =  _find_scale_factors_filtering(all_common_roots, xsyms, tol)
    return filter_scale_factors_by_projection_criteria(error_filtered,
                                                       frame1scaled,
                                                       frame2scaled)

def _find_scale_factors_filtering(all_common_roots, xsyms,  tol):
    all_common_roots_filtered = \
            _reject_errorneous(all_common_roots, tol=tol)

    all_common_roots_as_dict = [
        dict(s2root) for err, s2root in all_common_roots_filtered]
    rootarray = [[np.array(s2root[x], dtype=np.float64) for x in xsyms] 
                 for s2root in all_common_roots_as_dict] 
    errors = [err for err, s2root in all_common_roots_filtered]
    logger.debug("Solutions: {0}".format(all_common_roots_as_dict))
    logger.debug("errors: {0}".format(errors))
    # should we use tol=errors[0]?
    root_mat = _float_unique(rootarray, tol=tol)
    logger.debug("Got roots {0}".format(root_mat))
    return root_mat

def _reject_errorneous(all_common_roots, tol=TOL):
    # sort by error 
    all_common_roots_sorted = sorted(all_common_roots, key=lambda x:x[0])

    if all_common_roots_sorted[0][0] < tol:
        # we have at least one root error less than tolerance
        # filter by tolerance
        all_common_roots_sorted = [(err, s2root)
            for err, s2root in all_common_roots_sorted
            if err <= tol]
        temptol = tol
    #else:
    #    # reject all roots outside 10 times the error of most accurate root
    #    temptol = all_common_roots_sorted[0][0]
    #    all_common_roots_sorted = [(err, s2root)
    #        for err, s2root in all_common_roots_sorted
    #        if err <= 10*temptol]
    return all_common_roots_sorted

def _float_unique(rootarray, tol=TOL):
    if not len(rootarray):
        return []
    unique_roots = [rootarray[0]]
    for root in rootarray[1:]:
        close_root = _find_close_root(root, unique_roots, tol=tol)
        if close_root is None:
            unique_roots.append(root)
        else:
            unique_roots[close_root] = np.average(
                [unique_roots[close_root], root], axis=0)
    return unique_roots

def _find_close_root(root, unique_roots, tol=TOL):
    for r, uridx in it.product([root], range(len(unique_roots))):
        ur = unique_roots[uridx]
        if all([(abs(ri - uri) < tol) for ri, uri, in zip(r, ur)]):
            return uridx
    return None

def _solve_quad(quad):
    """
    >>> _solve_quad(x0**2 - 3*x0 + 2)
    [2, 1]
    """
    quad_coeffs = sympy.Poly(quad).all_coeffs()
    assert len(quad_coeffs) == 3, quad
    a2, a1, a0 = quad_coeffs
    term1 = -a1 
    term2 = sympy.sqrt(a1**2 - 4*a2*a0)
    return [(term1 + term2)/2*a2, (term1 - term2)/2*a2]

def _constraints(root, tol=TOL):
    # positive real part and imaginary part should be zero
    return sympy.re(root) > tol and abs(sympy.im(root)) < tol

def _verify_constraints(Pnum, sym2root, tol=TOL):
    # reject negative roots
    return all(_constraints(root) for sym, root in sym2root)

def _compute_error(Pnum, sym2root, tol=TOL):
    return sum(abs(eq.subs(dict(sym2root))) for eq in Pnum)

def _find_common_roots(Pnum, roots, sym, sym2polyidx, sympair2polyidx, tol=TOL):
    # TODO: get rid of all sympy calculations by using lambdification to make it faster
    filtered_roots = list()
    eqns = sym2polyidx[sym]
    for r in roots:
        sym2roots = list()
        # we already know that rest of the equations are going to be
        # quadratics with 2 roots. To have the same number of roots throughout
        # we add roots for this symbol twice
        sym2roots.append(((sym, r), (sym ,r))) # a bad data structure
        for eqidx in sym2polyidx[sym]:
            eq = Pnum[eqidx]
            quad = eq.subs({sym : r })
            s = poly_free_symbols(quad).pop() # i know it's one
            sym2roots.append([(s, qr) for qr in _solve_quad(quad)])

        # sym2roots is a list with n-elments where n is the number of
        # variables. Each element is the root list corresponding to one symbol
        for s2root in it.product(*sym2roots):
            if _verify_constraints(Pnum, s2root, tol=tol):
                s2root = [(s, sympy.re(r)) for s, r in s2root]
                err = _compute_error(Pnum, s2root, tol=tol)
                filtered_roots.append((err, s2root))

    return filtered_roots

def p3p(pis, uis):
    p1, p2, p3 = pis
    u1, p2, u3 = uis


