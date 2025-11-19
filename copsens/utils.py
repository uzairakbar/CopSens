import numpy as np
import cvxpy as cp

def get_opt_gamma(mu_y_dt, mu_u_dt, cov_u_t, sigma_y_t, R2_constr=1.0, normtype="L2", idx=None):
    """
    Obtain Optimized Sensitivity Parameters Using Multivariate Calibration Criterion.
    Translates get_opt_gamma.R logic to Python using cvxpy.
    """
    n_confounders = mu_u_dt.shape[1]
    gamma = cp.Variable(n_confounders)
    
    if idx is None:
        idx = np.ones(len(mu_y_dt))
    
    # Define Objective
    residuals = cp.multiply(idx, (mu_y_dt - mu_u_dt @ gamma))
    
    if normtype == "L1":
        obj = cp.norm1(residuals)
    elif normtype == "L2":
        obj = cp.norm2(residuals)
    elif normtype == "Inf":
        obj = cp.norm_inf(residuals)
    else:
        raise ValueError("normtype must be 'L1', 'L2', or 'Inf'")
    
    # Define Constraint
    # R code: t(gamma) %*% cov_u_t %*% gamma <= R2 * sigma_y^2
    # Note: cov_u_t / sigma_y_t^2 scaling happens in constraint
    quad_form = cp.quad_form(gamma, cov_u_t)
    constraint = [quad_form <= R2_constr * (sigma_y_t**2)]
    
    prob = cp.Problem(cp.Minimize(obj), constraint)
    try:
        prob.solve()
    except Exception as e:
        print(f"Optimization failed: {e}")
        return np.zeros(n_confounders)

    if gamma.value is None:
        return np.zeros(n_confounders)
        
    return gamma.value

def cal_rv(mu_y_dt, sigma_y_t, mu_u_dt, cov_u_t):
    """
    Calculate Robustness Value (RV).
    """
    # Calculate inverse square root of covariance
    # Python's eigh returns eigenvalues in ascending order
    vals, vecs = np.linalg.eigh(cov_u_t) 
    # Filter small values for numerical stability
    vals = np.maximum(vals, 1e-10)
    cov_halfinv = vecs @ np.diag(vals**-0.5) @ vecs.T
    
    # Calculate numerator term: magnitude of confounding needed
    # apply(mu_u_dt %*% cov_halfinv, 1, function(x) sum(x^2))
    transformed_mu = mu_u_dt @ cov_halfinv
    denom_confounding = np.sum(transformed_mu**2, axis=1)
    
    rv = (mu_y_dt**2) / denom_confounding / (sigma_y_t**2)
    
    # RV > 1 implies the effect is robust to any amount of confounding explained by R2=1
    rv_out = rv.copy()
    rv_out[rv_out > 1] = np.nan 
    return rv_out
