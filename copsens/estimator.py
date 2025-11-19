import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import norm
import warnings

from .utils import get_opt_gamma, cal_rv
from .nn_utils import fit_torch_model, get_torch_embeddings

class CopSens(BaseEstimator):
    """
    Copula-based Sensitivity Analysis for Multi-Treatment Causal Inference.
    
    Parameters
    ----------
    model_type : str, default='gaussian'
        Type of outcome variable. Options: 'gaussian', 'binary'.
    latent_model : str or torch.nn.Module, default='pca'
        Method to learn latent confounders. 'pca' uses Probabilistic PCA logic via Sklearn.
        Can pass a PyTorch nn.Module (VAE) instance or class.
    n_components : int, default=3
        Number of latent confounders.
    """
    def __init__(self, model_type='gaussian', latent_model='pca', n_components=3, verbose=False):
        self.model_type = model_type
        self.latent_model = latent_model
        self.n_components = n_components
        self.verbose = verbose
        
        # Fitted artifacts
        self.pca_model_ = None
        self.nn_model_ = None
        self.outcome_model_ = None
        self.sigma_y_t_ = None # Residual SD for Gaussian
        
        # Latent stats storage
        self.mu_u_tr_ = None # Latent scores for training data
        self.cov_u_t_ = None # Conditional covariance matrix
        
    def fit(self, T, y):
        """
        Fit the latent variable model on Treatments (T) and the outcome model on (T, y).
        
        Parameters
        ----------
        T : array-like of shape (n_samples, n_features)
            Treatment variables.
        y : array-like of shape (n_samples,)
            Outcome variable.
        """
        T = np.array(T)
        y = np.array(y)
        
        # 1. Fit Latent Confounder Model (T -> U)
        if self.verbose: print(f"Fitting latent model using {self.latent_model}...")
        
        if self.latent_model == 'pca':
            self.pca_model_ = PCA(n_components=self.n_components)
            self.mu_u_tr_ = self.pca_model_.fit_transform(T)
            
            # PPCA Covariance Estimation logic:
            # sigma^2_est (noise variance)
            sigma2_est = self.pca_model_.noise_variance_
            W = self.pca_model_.components_.T # Loadings (n_features, n_components)
            
            # cov(U|t) = sigma2 * (W'W + sigma2*I)^-1
            # For PCA, W columns are orthogonal, but let's use the full formula
            M = W.T @ W + sigma2_est * np.eye(self.n_components)
            self.cov_u_t_ = sigma2_est * np.linalg.inv(M)
            
        else:
            # Assume PyTorch Model
            self.nn_model_ = fit_torch_model(T, self.n_components, user_model=self.latent_model)
            self.mu_u_tr_, self.cov_u_t_ = get_torch_embeddings(self.nn_model_, T)

        # 2. Fit Outcome Model (T -> Y)
        if self.verbose: print(f"Fitting observed outcome model ({self.model_type})...")
        
        if self.model_type == 'gaussian':
            self.outcome_model_ = LinearRegression()
            self.outcome_model_.fit(T, y)
            y_pred = self.outcome_model_.predict(T)
            # Estimate residual standard deviation
            self.sigma_y_t_ = np.sqrt(np.sum((y - y_pred)**2) / (len(y) - T.shape[1] - 1))
            
        elif self.model_type == 'binary':
            # Use Statsmodels Probit for accuracy matching R
            # Add constant for intercept
            T_const = sm.add_constant(T, has_constant='add')
            self.outcome_model_ = sm.Probit(y, T_const).fit(disp=0)
            
        return self

    def predict(self, T):
        """
        Returns the naive (uncalibrated) outcome predictions.
        """
        T = np.array(T)
        if self.model_type == 'gaussian':
            return self.outcome_model_.predict(T)
        elif self.model_type == 'binary':
            T_const = sm.add_constant(T, has_constant='add')
            return self.outcome_model_.predict(T_const) # Returns probabilities

    def calibrate(self, t1, t2, calitype='worstcase', R2=None, R2_constr=1.0, gamma=None, normtype='L2'):
        """
        Perform Sensitivity Analysis.
        
        Parameters
        ----------
        t1 : array-like
            Treatment condition 1 (Target).
        t2 : array-like
            Treatment condition 2 (Control/Baseline).
        calitype : str
            'worstcase', 'multicali', or 'null'.
        R2 : list or float
            Partial R2 values for worstcase/null calibration.
        R2_constr : float
            Constraint for multicali.
        gamma : array-like
            User specified gamma for 'null' calibration.
            
        Returns
        -------
        dict containing 'est_df', 'gamma', 'R2', 'rv'
        """
        t1 = np.array(t1)
        t2 = np.array(t2)
        
        # 1. Get Naive Estimates (Difference)
        if self.model_type == 'gaussian':
            mu_y_t1 = self.outcome_model_.predict(t1)
            mu_y_t2 = self.outcome_model_.predict(t2)
            mu_y_dt = mu_y_t1 - mu_y_t2
        else:
            T1_const = sm.add_constant(t1, has_constant='add')
            T2_const = sm.add_constant(t2, has_constant='add')
            mu_y_t1 = self.outcome_model_.predict(T1_const)
            mu_y_t2 = self.outcome_model_.predict(T2_const)
            # For binary, we calibrate the probability, but 'mu_y_dt' is used for RV/optimization
            # We handle binary calibration via simulation later
            mu_y_dt = mu_y_t1 - mu_y_t2 

        # 2. Get Latent Variable Differences
        if self.latent_model == 'pca':
            u_t1 = self.pca_model_.transform(t1)
            u_t2 = self.pca_model_.transform(t2)
        else:
            u_t1, _ = get_torch_embeddings(self.nn_model_, t1)
            u_t2, _ = get_torch_embeddings(self.nn_model_, t2)
        
        mu_u_dt = u_t1 - u_t2
        
        # Pre-calculate covariance half-inverse for worst-case bounds
        vals, vecs = np.linalg.eigh(self.cov_u_t_)
        vals = np.maximum(vals, 1e-10)
        cov_halfinv = vecs @ np.diag(vals**-0.5) @ vecs.T
        
        results = {}
        
        # --- GAUSSIAN OUTCOME LOGIC ---
        if self.model_type == 'gaussian':
            
            if calitype == 'worstcase':
                if R2 is None: R2 = [1.0]
                R2 = np.array(R2)
                
                # Calculate Robustness Value
                rv = cal_rv(mu_y_dt, self.sigma_y_t_, mu_u_dt, self.cov_u_t_)
                
                # Calculate Bias Bounds
                # Bias magnitude = sigma_y * ||cov^{-1/2} @ mu_u_dt|| * sqrt(R2)
                u_norm = np.sqrt(np.sum((mu_u_dt @ cov_halfinv)**2, axis=1))
                
                est_data = {'Naive': mu_y_dt}
                
                for r2_val in R2:
                    bias_mag = self.sigma_y_t_ * u_norm * np.sqrt(r2_val)
                    est_data[f'R2_{r2_val}_lwr'] = mu_y_dt - bias_mag
                    est_data[f'R2_{r2_val}_upr'] = mu_y_dt + bias_mag
                    
                results['est_df'] = pd.DataFrame(est_data)
                results['rv'] = rv
                results['R2'] = R2
                
            elif calitype == 'multicali':
                # Optimize gamma
                opt_gamma = get_opt_gamma(mu_y_dt, mu_u_dt, self.cov_u_t_, 
                                          self.sigma_y_t_, R2_constr, normtype)
                
                bias = mu_u_dt @ opt_gamma
                calibrated = mu_y_dt - bias
                
                results['est_df'] = pd.DataFrame({'Naive': mu_y_dt, 'Calibrated': calibrated})
                results['gamma'] = opt_gamma
                results['R2_implied'] = (opt_gamma.T @ self.cov_u_t_ @ opt_gamma) / (self.sigma_y_t_**2)

            elif calitype == 'null':
                if gamma is None: raise ValueError("gamma must be provided for calitype='null'")
                gamma = np.array(gamma)
                if R2 is None: R2 = [1.0]
                
                est_data = {'Naive': mu_y_dt}
                # Note: if gamma is direction, we scale by R2. 
                # R Logic: gamma = sqrt(R2) * gamma / sqrt(gamma' cov gamma)
                
                denom = np.sqrt(gamma.T @ self.cov_u_t_ @ gamma)
                if denom < 1e-9: denom = 1.0
                
                gamma_norm = gamma / denom
                
                for r2_val in R2:
                     scaled_gamma = np.sqrt(r2_val) * gamma_norm
                     bias = self.sigma_y_t_ * mu_u_dt @ cov_halfinv @ scaled_gamma
                     # Wait, R logic: mu_y - sqrt(R2) * sigma * mu_u_dt %*% cov_halfinv %*% gamma_norm
                     # Actually simpler: Bias = mu_u_dt @ (True Gamma).
                     # If user provides direction d, gamma = sigma * sqrt(R2) * cov^-0.5 * d
                     
                     # Using the equation (37) from paper logic implemented in R's gcalibrate null:
                     bias = np.sqrt(r2_val) * self.sigma_y_t_ * (mu_u_dt @ cov_halfinv @ gamma_norm)
                     est_data[f'R2_{r2_val}'] = mu_y_dt - bias
                
                results['est_df'] = pd.DataFrame(est_data)

        # --- BINARY OUTCOME LOGIC ---
        elif self.model_type == 'binary':
            # For binary, we simulate latent confounders to adjust probability
            # This corresponds to `cali_mean_ybinary_algm.R`
            
            if gamma is None:
                 # If gamma not provided (e.g. worstcase), we need to infer a direction
                 # Usually worstcase isn't closed form for binary. 
                 # We assume 'null' calibration logic where user provides gamma or we pick one.
                 if calitype == 'worstcase':
                      # Pick direction maximizing mu_u difference? 
                      # Simplified: defaulting to null with arbitrary direction is risky.
                      # For this impl, we require gamma for binary or default to naive.
                      raise NotImplementedError("Worst-case closed form not available for Binary. Use 'null' with specific gamma.")

            nsim = 1000
            gamma = np.array(gamma)
            
            # Normalize gamma direction
            denom = np.sqrt(gamma.T @ self.cov_u_t_ @ gamma)
            if denom < 1e-9: denom = 1.0
            gamma_dir = gamma / denom # Unit direction in confounder space
            
            if R2 is None: R2 = [0.1]
            
            est_data = {'Naive': mu_y_t1} # Comparing to t1 outcomes usually
            
            # mu_u_t corresponds to the latent of the target treatment
            # We simulate for every observation
            
            # Pre-generate noise: (N_samples, N_sim)
            noise = np.random.normal(0, 1, (len(t1), nsim))
            
            for r2_val in R2:
                # Scale gamma: gamma_scaled such that it explains R2 variance of latent effect
                # In probit: Y* = T beta + U gamma + eps. Var(eps)=1. 
                # This is complex. Following R implementation logic:
                # gamma vector is scaled. 
                
                # R code: gamma <- sqrt(R2[i]) * gamma / sqrt(c(t(gamma) %*% cov_u_t %*% gamma))
                # This scales gamma such that variance contribution is related to R2?
                # Actually in R bcalibrate:
                # mu_i <- (mu_u_tr - mu_u_t[i,]) %*% gamma
                # ytilde = N(mu_i, 1)
                # prob = mean( pnorm(ytilde) > 1 - mu_y_t )
                
                gamma_final = np.sqrt(r2_val) * gamma_dir 
                
                calibrated_probs = []
                
                # Vectorized simulation
                # mu_shift: Influence of confounder difference between observed TR and hypothetical T
                # But here we are estimating E[Y | do(t)].
                # R code uses `mu_u_tr` (observed latent) and `mu_u_t` (counterfactual latent).
                
                # Shift = (U_obs - U_cf) @ gamma
                shift = (self.mu_u_tr_ - u_t1) @ gamma_final
                
                # Simulation
                # y_tilde = shift + noise
                y_tilde_samples = shift[:, None] + noise
                
                # Threshold: pnorm(y_tilde) > 1 - naive_prob
                # For probit, pnorm is CDF.
                # R: y_samples <- ifelse(pnorm(ytilde_samples) > 1 - mu_y_t[i], 1, 0)
                
                # Inverse Probit (Quantile)
                thresh = norm.ppf(mu_y_t1) # Threshold on latent scale
                
                # If y_tilde > threshold? 
                # R logic is slightly different, it acts on p-values.
                # Let's follow R exactly:
                # mu_y_t is probability. 1 - mu_y_t is prob(0).
                thresholds = 1 - mu_y_t1
                
                # prob of success
                probs = norm.cdf(y_tilde_samples)
                successes = (probs > thresholds[:, None]).astype(float)
                calibrated_probs = np.mean(successes, axis=1)
                
                est_data[f'R2_{r2_val}'] = calibrated_probs
                
            results['est_df'] = pd.DataFrame(est_data)
            
        return results
