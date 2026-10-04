"""
Black-Scholes analytical formulas, Greeks, and utility functions
"""
import math
import numpy as np
import torch
from scipy.stats import norm


def black_scholes_price(S, K, T, sigma, r, option_type='call'):
    """Black-Scholes analytical price for European options. T in years; S/K/T/sigma/r may be scalars or broadcastable arrays."""
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    S, K, T, sigma, r = np.broadcast_arrays(*[
        np.asarray(value, dtype=float) for value in (S, K, T, sigma, r)
    ])
    if not all(np.isfinite(value).all() for value in (S, K, T, sigma, r)):
        raise ValueError('Pricing inputs must be finite')
    if np.any(S < 0) or np.any(K <= 0) or np.any(T < 0) or np.any(sigma < 0):
        raise ValueError('Require S >= 0, K > 0, T >= 0 and sigma >= 0')
    discount_strike = K * np.exp(-r * T)
    deterministic = np.maximum(S - discount_strike, 0.0) if option_type == 'call' else np.maximum(discount_strike - S, 0.0)
    regular = (T > 0) & (sigma > 0) & (S > 0)
    price = np.array(deterministic, copy=True)
    price[regular] = _bs_price(S[regular], K[regular], T[regular], sigma[regular], r[regular], option_type)
    return price.item() if price.ndim == 0 else price


def _bs_price(S, K, T, sigma, r, option_type):
    """Internal helper for BS price (assumes T > 0)"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_greeks(S, K, T, sigma, r, option_type='call'):
    """
    Compute Greeks for European options.
    
    Returns
    -------
    greeks : dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    if T <= 0:
        # At expiry: Greeks are not well-defined (discontinuous)
        delta = np.where(S > K, 1.0, 0.0) if option_type == 'call' else np.where(S > K, 0.0, -1.0)
        return {
            'delta': delta,
            'gamma': np.zeros_like(S),
            'vega': np.zeros_like(S),
            'theta': np.zeros_like(S),
            'rho': np.zeros_like(S)
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
    
    # Gamma (same for call and put)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    
    # Vega (same for call and put)
    vega = S * pdf_d1 * sqrt_T / 100  # Divided by 100 for 1% change
    
    # Theta
    common_theta = -S * pdf_d1 * sigma / (2 * sqrt_T)
    if option_type == 'call':
        theta = common_theta - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = common_theta + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = theta / 365  # Per-day theta
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def monte_carlo_price(S, K, T, sigma, r, n_paths=100000, seed=42, option_type='call'):
    """Monte Carlo price for a European option. Returns (price, std_err)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    
    discount = np.exp(-r * T)
    prices = discount * payoffs
    
    return np.mean(prices), np.std(prices) / np.sqrt(n_paths)



def _normal_cdf(x):
    """Standard normal CDF using erf (torch.normal_cdf does not exist)."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def black_scholes_price_torch(S, K, T, sigma, r, option_type='call'):
    """
    PyTorch version of Black-Scholes pricing (supports batching on GPU).
    """
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    eps = 1e-10
    sqrt_T = torch.sqrt(T.clamp(min=eps))

    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == 'call':
        price = S * _normal_cdf(d1) - K * torch.exp(-r * T) * _normal_cdf(d2)
    else:
        price = K * torch.exp(-r * T) * _normal_cdf(-d2) - S * _normal_cdf(-d1)
    
    # Handle T ≈ 0 case
    at_expiry = T <= eps
    if option_type == 'call':
        payoff_at_expiry = torch.maximum(S - K, torch.tensor(0.0, device=S.device))
    else:
        payoff_at_expiry = torch.maximum(K - S, torch.tensor(0.0, device=S.device))
    
    price = torch.where(at_expiry, payoff_at_expiry, price)
    return price


def compute_rmse(pred, true):
    """Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - true)**2))


def compute_mape(pred, true, eps=1e-8):
    """Mean Absolute Percentage Error"""
    mask = true.abs() > eps
    if not torch.any(mask):
        return pred.new_tensor(float('nan'))  # No nonzero targets: MAPE is undefined.
    return torch.mean(torch.abs((pred[mask] - true[mask]) / true[mask])) * 100


def compute_max_error(pred, true):
    """Maximum Absolute Error"""
    return torch.max(torch.abs(pred - true))



def upload_to_gcp_bucket(local_file_path, bucket_name, destination_blob_name, service_account_json_path, atomic=True, storage_class='STANDARD'):
    """
    Upload a file to GCS using a service account.
    atomic=True uploads to a temp blob first to prevent partial overwrites.
    storage_class: 'STANDARD', 'NEARLINE', 'COLDLINE', or 'ARCHIVE'.
    """
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        import os

        if not os.path.exists(local_file_path):
            print(f"File not found: {local_file_path}")
            return False

        credentials = service_account.Credentials.from_service_account_file(service_account_json_path)
        client = storage.Client(credentials=credentials, project=credentials.project_id)
        bucket = client.bucket(bucket_name)

        if atomic:
            temp_blob_name = destination_blob_name + '.tmp'
            temp_blob = bucket.blob(temp_blob_name)
            temp_blob.storage_class = storage_class
            print(f"Uploading {local_file_path} to gs://{bucket_name}/{temp_blob_name} (temp) [{storage_class}]...")
            temp_blob.upload_from_filename(local_file_path)

            # Rename: Bucket.rename_blob(old_blob, new_name) — not Blob.rename()
            bucket.rename_blob(temp_blob, destination_blob_name)
            print(f"✓ Upload complete: gs://{bucket_name}/{destination_blob_name}")
        else:
            blob = bucket.blob(destination_blob_name)
            blob.storage_class = storage_class
            print(f"Uploading {local_file_path} to gs://{bucket_name}/{destination_blob_name} [{storage_class}]...")
            blob.upload_from_filename(local_file_path)
            print(f"✓ Upload complete.")

        return True
    except ImportError:
        print("google-cloud-storage package is not installed. Run: pip install google-cloud-storage")
        return False
    except Exception as e:
        print(f"Failed to upload to GCP: {e}")
        return False

