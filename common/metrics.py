import numpy as np
from scipy import linalg


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (as in FID / FAD)."""
    mu_a, mu_b = features_a.mean(0), features_b.mean(0)
    cov_a, cov_b = np.cov(features_a, rowvar=False), np.cov(features_b, rowvar=False)
    covmean = linalg.sqrtm(cov_a @ cov_b).real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))
