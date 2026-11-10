import librosa
import numpy as np
import soundfile as sf
import pywt

from spectrogram.compare_spectrograms import compare_spectrograms

def main():
    print("Start of program")
    file_path = "data/raw/xeno_canto/Song_Sparrow/XC1090712 - Song Sparrow - Melospiza melodia.mp3"
    write_audio_path = "outputs/Song_Sparrow/XC1090712 - Song Sparrow - Melospiza melodia_wavelet_threshold.wav"
    write_spectrogram_path = "outputs/Song_Sparrow/XC1090712 - Song Sparrow - Melospiza melodia_wavelet_threshold_spectrogram.png"

    audio, sr = librosa.load(file_path, sr=None)

    reduced_audio = epanechnikov_shrink(audio)

    sf.write(write_audio_path, reduced_audio, sr)
    print(f"Saved reduced noise to {write_audio_path}")

    compare_spectrograms(audio, reduced_audio, sr, write_to_path=write_spectrogram_path)

    print("End of program")

def calculate_alpha(
        j: int,
        J0: int = 0,
        l: int = 1,
        gamma: float = 2.0
    ) -> float:
    """
    Calculation for the alpha hyperparameter.

    Represents level-dependent spike probability.

    Args:
        j: Wavelet decomposition level.
        J0: Starting decomposition level.
        l: Must be a positive integer.
        gamma: Base of the exponential decay.

    Returns:
        Calculated alpha hyperparameter.
    """

    return 1.0 - 1.0 / (j - J0 + l) ** gamma

# utility function for hyperparameter beta elicitation
def calculate_beta(detail):
    """
    Calculation for the beta hyperparameter.

    Represents the Epanechnikov prior support parameter.

    Args:
        detail: Wavelet detail coefficients.

    Returns:
        Calculated beta hyperparameter.
    """
    return float(np.max(np.abs(detail)))

def epanechnikov_density(
    theta: float,
    beta: float
    ) -> float:
    """
    Continuous Epanechnikov component of the prior
    """

    if abs(theta) > beta:
        return 0.0

    return (3.0 / (4.0 * (beta ** 3)) * (beta ** 2 - theta ** 2))   # missing I application?

def calculate_lambda(
    finest_detail: np.ndarray,
    c: float = 1.0,
    tau: float = 2.0
) -> float:
    """
    Calculates the noise-variance prior hyperparameter.

    Paper: Equation (11).

    s is the sample standard deviation of the
    finest-resolution detail coefficients.
    """
    finest_detail = np.asarray(finest_detail, dtype=np.float64)

    if finest_detail.ndim != 1 or finest_detail.size < 2:
        raise ValueError("Finest detail must contain at least two coefficients.")

    if not np.all(np.isfinite(finest_detail)):
        raise ValueError("Detail coefficients must be finite.")

    if c <= 0 or tau <= 0:
        raise ValueError("c and tau must be positive.")

    s = np.std(finest_detail, ddof=1)

    if s <= 0 or not np.isfinite(s):
        raise ValueError("Finest detail must have positive standard deviation.")

    return float(
        1.0 / s**2 +
        (c / tau) * np.exp(-s / tau)
    )


def epanechnikov_posterior_mean(
    detail: np.ndarray,
    alpha: float,
    beta: float,
    lambda_value: float
) -> np.ndarray:
    """
    Calculates the posterior mean for observed wavelet coefficients.

    Paper: Equation (8).

    Uses the spike-and-slab Epanechnikov prior and
    exponential prior on the noise variance.
    """
    detail = np.asarray(detail, dtype=np.float64)

    if not np.all(np.isfinite(detail)):
        raise ValueError("Detail coefficients must be finite.")

    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1).")

    if beta <= 0 or lambda_value <= 0:
        raise ValueError("beta and lambda must be positive.")

    # All coefficients must be within the prior support.
    if np.any(np.abs(detail) > beta * (1.0 + 1e-10)):
        raise ValueError("Coefficients exceed the Epanechnikov support.")

    # Degenerate prior: all coefficients are zero.
    if beta == 0:
        return np.zeros_like(detail)

    d = detail
    lam = lambda_value
    root = np.sqrt(2.0 * lam)

    # Exponential terms in Equation (8).
    exp_minus = np.exp(-root * (beta - d))
    exp_plus = np.exp(-root * (beta + d))

    # Numerator of the posterior mean.
    numerator_bracket = (
        (
            2.0 * lam * beta**2
            + 3.0 * root * beta
            + 3.0
        ) / (2.0 * lam**2)
    ) * (exp_minus - exp_plus)

    numerator_bracket += (
        (lam * beta**2 - 3.0) * root * d
        - lam * root * d**3
    ) / lam**2

    numerator = (
        (1.0 - alpha)
        * (3.0 * root / (8.0 * beta**3))
        * numerator_bracket
    )

    # Double-exponential density associated with the spike.
    spike_density = (
        root / 2.0
    ) * np.exp(-root * np.abs(d))

    # Continuous slab contribution to the denominator.
    slab_bracket = (
        (beta / lam) * (exp_plus + exp_minus)
        + (2.0 / root) * (
            beta**2 - d**2 - 1.0 / lam
        )
    )

    slab_density = (
        (1.0 - alpha)
        * (3.0 * root / (8.0 * beta**3))
        * slab_bracket
    )

    denominator = (
        alpha * spike_density
        + slab_density
    )

    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError(
            "Invalid posterior denominator. Check numerical stability."
        )

    result = numerator / denominator

    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Posterior mean produced non-finite values.")

    return result


def epanechnikov_shrink(
    audio: np.ndarray,
    wavelet: str = "db10",
    level: int = 5,
    mode: str = "symmetric",
    J0: int = 0,
    l: int = 1,
    gamma: float = 2.0,
    c: float = 1.0,
    tau: float = 2.0
) -> np.ndarray:
    """
    Denoise an audio signal using wavelet-based Bayesian shrinkage
    with the Epanechnikov prior.

    Based off of the technique given in "Processing and classifying bird songs using wavelet
    techniques and supervised learning,"
    
    Laura Luca Dominguez Barrios, Fidel Aniano Causil Barrios, Alex Rodrigo dos Santos Sousa,
    and Marianna Rodrigues Motta; all from the Department of Statistics, State University of
    Campinas, Campinas, Brazil.

    arXiv:2609.10826

    Args:
        audio: Input audio waveform.
        wavelet: PyWavelets wavelet identifier.
        level: Number of DWT decomposition levels.
        mode: Wavelet boundary extension mode.
        J0: Primary resolution level.
        l: Alpha hyperparameter.
        gamma: Alpha hyperparameter.
        c: Lambda hyperparameter.
        tau: Lambda hyperparameter.

    Returns:
        Denoised audio waveform.
    """
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1:
        raise ValueError("Audio must be one-dimensional.")

    if audio.size == 0:
        raise ValueError("Audio cannot be empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio must contain only finite values.")

    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(audio), wav.dec_len)

    if level < 1 or level > max_level:
        raise ValueError(
            f"Invalid decomposition level. Maximum supported level: {max_level}"
        )

    # 1. Wavelet decomposition.
    coeffs = pywt.wavedec(
        audio,
        wavelet=wavelet,
        level=level,
        mode=mode
    )

    # PyWavelets ordering:
    # [cA_J, cD_J, cD_(J-1), ..., cD_1]

    # 2. Estimate lambda once using the finest detail coefficients.
    finest_detail = coeffs[-1]
    lambda_value = calculate_lambda(
        finest_detail,
        c=c,
        tau=tau
    )

    # 3. Preserve the approximation coefficients.
    denoised_coeffs = [coeffs[0]]

    # 4. Shrink each detail subband.
    for i, detail in enumerate(coeffs[1:]):

        # Paper indexing:
        # cD_J -> j=J0
        # cD_1 -> j=J0 + level - 1
        j = J0 + i

        alpha = calculate_alpha(
            j=j,
            J0=J0,
            l=l,
            gamma=gamma
        )

        beta = calculate_beta(detail)

        denoised_detail = epanechnikov_posterior_mean(
            detail=detail,
            alpha=alpha,
            beta=beta,
            lambda_value=lambda_value
        )

        denoised_coeffs.append(denoised_detail)

    # 5. Reconstruct the waveform.
    denoised_audio = pywt.waverec(
        denoised_coeffs,
        wavelet=wavelet,
        mode=mode
    )

    return denoised_audio[:len(audio)]




if __name__ == "__main__":
    main()