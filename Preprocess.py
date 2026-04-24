import numpy as np

# -------- Amplitude --------
def normalize_amplitude(amp):
    min_val = np.min(amp)
    max_val = np.max(amp)
    return (amp - min_val) / (max_val - min_val + 1e-8)

# -------- Phase --------
def unwrap_phase(phase):
    return np.unwrap(phase, axis=0)

# -------- Gaussian Encoding --------
def gaussian_encoding(x, sigma=1.0):
    return np.exp(-(x**2) / (2 * sigma**2))

# -------- Main pipeline --------
def preprocess_csi(csi):
    """csi shape: (time, subcarriers, 2)
    2 = [amplitude, phase]
    """
    amp = csi[:, :, 0]
    phase = csi[:, :, 1]
    
    # Process
    amp = normalize_amplitude(amp)
    phase = unwrap_phase(phase)
    
    # Encoding
    amp = gaussian_encoding(amp)
    phase = gaussian_encoding(phase)
    
    return amp, phase