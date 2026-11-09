---------------- MODULE Pipeline ----------------

(* Pipeline for modeling a Bird Song denoising pipeline.                        *)
(*                                                                              *)
(* The goal of the system is to learn to denoise bird songs from a given dataset*)
(* (such as xenocanto). Our method relies on analyzing the time and frequency   *)
(*                                                                              *)
(* components of the signal to identify noise.                                  *)
(* The pipeline receives as input:                                              *)
(*   - A an initial time-domain recording with noise x_0(t)                     *)
(*                                                                              *)
(* The pipeline operates as follows:                                            *)
(*                                                                              *)
(*      1. INITIALIZATION: the time-domain signal x_0(t) is received            *)
(*                                                                              *)
(*      2. PREPROCESSING:                                                       *)
(*         - STANDARDIZATION ~ x(t)                                             *)
(*              The signal is resampled to a prespecified rate                  *)
(*                                                                              *)
(*         - STFT ~ X(t, f)                                                     *)
(*              The signal is transformed to the time-frequency domain          *)
(*                                                                              *)
(*         - Phase/Magnitude Split ~ M(t, f), PHI(t, f)                         *)
(*              Splits the signal into magnitude and phase components, which are*)
(*              required as arguments for later methods.                        *)
(*                                                                              *)
(*      3. ESTABLISHING A BASELINE: Use non-machine learning methods for        *)
(*        denoising to compare effectiveness of the machine learning methods.   *)
(*                                                                              *)
(*         - Heuristic Spectral Gating ~ X_clean(t, f)                          *)
(*              Removes noise by subtracting the estimated noise spectrum from  *)
(*              the observed spectrum.                                          *)
(*                                                                              *)
(*              Requires Magnitude M(t,f), Phase PHI(t,f),                      *)
(*              and an estimate noise profiler N(f).                            *)
(*                                                                              *)
(*      4. DENOISING: Extract a denoising magnitude matrix from the noisy       *)
(*        matrix M(t, f) produced from the preprocessing.                       *)
(*                                                                              *)
(* More preceisely, the system should follow the following state sequence:      *)
(*                                                                              *)
(*      init -> standardized -> time-frequency -> phase-split -> baseline       *)
(*      -> denoising                                                            *)
(*                                                                              *)


EXTENDS Integers, Naturals, FiniteSets

CONSTANTS
    StateSet,       \* represent set of main high-level states
    Unassigned,     \* represents an unassigned state
    T,              \* represent type for time-domain arguments
    F,              \* represent type for frequency-domain arguments
    AmplitudeValues,\* represent set of amplitude values
    TargetSampleRate \* represent input sample rate

ASSUME T # {}
ASSUME F # {}
ASSUME AmplitudeValues # {}
ASSUME TargetSampleRate \in Nat
ASSUME Unassigned \notin StateSet

Times == 0..1               \* abstraction for time-domain steps
Frequencies == { 1000 }     \* abstraction for frequency-domain Hz measurements
Amplitudes == { -1, 1 }     \* abstraction for amplitude values

VARIABLES
    state,          \* track main high-level states
    x_0,            \* initial input signal, maps time-domain arguments to amplitudes
    x,              \* standardized signal, maps time-domain arguments to amplitudes
    X,              \* time-frequency representation, maps time-domain arguments to frequency-domain amplitudes
    M,              \* magnitude matrix, maps time-domain arguments to frequency-domain amplitudes
    PHI,            \* phase matrix, maps time-domain arguments to frequency-domain phases
    B,              \* baseline matrix, maps time-domain arguments to frequency-domain amplitudes
    sampleRate      \* track's sample rate


vars == << state, x_0, x, X, M, PHI, B, sampleRate >>

\* Abstraction for sample-rate trait for audio tracks, maps input to a sample rate

Init ==
    /\ state = "init"
    /\ x_0 \in [ T -> AmplitudeValues ]     \* initial audio signal
    /\ x = Unassigned
    /\ X = Unassigned
    /\ M = Unassigned
    /\ PHI = Unassigned
    /\ B = Unassigned
    /\ sampleRate \in (TargetSampleRate - 1)..(TargetSampleRate + 1)    \* sample rates vary

\* init -> standardized
\* Abstraction for standardization of the input dataset,
\* which resamples it to a consistent sample rate.
Standardize ==
    /\ state = "init"
    /\ x = Unassigned
    /\ sampleRate' = TargetSampleRate   \* normalize sample rate
    /\ x' = x_0                         \* resample input signal
    /\ state' = "standardized"
    /\ UNCHANGED << x_0, X, M, PHI, B >>

\* standardized -> time-frequency
\* Abstraction for time-frequency analysis of the standardized signal.
\* Simulate Short-Time Fourier Transform (STFT)
STFT ==
    /\ state = "standardized"
    /\ X = Unassigned
    /\ X' \in [ T -> [ F -> Frequencies ] ]
    /\ state' = "time-frequency"
    /\ UNCHANGED << x_0, x, M, PHI, B, sampleRate >>

\* time-frequency -> phase-split
\* Abstraction for phase splitting of the time-frequency representation.
PhaseSplit ==
    /\ state = "time-frequency"
    /\ M = Unassigned
    /\ PHI = Unassigned
    /\ M' \in [ T -> [ F -> AmplitudeValues ] ]
    /\ PHI' \in [ T -> [ F -> AmplitudeValues ] ]
    /\ state' = "phase-split"
    /\ UNCHANGED << x_0, x, X, B, sampleRate >>

\* phase-split -> baseline
\* Abstraction for baseline estimation of the phase-split representation.
BaseLine ==
    /\ state = "phase-split"
    /\ B = Unassigned
    /\ B' \in [ T -> [ F -> AmplitudeValues ] ]
    /\ state' = "baseline"
    /\ UNCHANGED << x_0, x, X, M, PHI, sampleRate >>

\* baseline -> denoising
\* Abstraction for denoising of the standardized (not denoised) representation.
Denoise ==
    /\ state = "baseline"
    /\ x' \in [ T -> AmplitudeValues ]
    /\ state' = "denoising"
    /\ UNCHANGED << x_0, X, M, PHI, B, sampleRate >>

Next ==
    \/ Standardize
    \/ STFT
    \/ PhaseSplit
    \/ BaseLine
    \/ Denoise
    \/ UNCHANGED << vars >>

Spec == Init /\ [][Next]_vars

TypeOk ==
    /\ state \in StateSet
    /\ x_0 \in [ T -> AmplitudeValues ]
    /\ x \in [ T -> AmplitudeValues ] \cup { Unassigned }
    /\ X \in [ T -> [ F -> Frequencies ] ] \cup { Unassigned }
    /\ M \in [ T -> [ F -> AmplitudeValues ] ] \cup { Unassigned }
    /\ PHI \in [ T -> [ F -> AmplitudeValues ] ] \cup { Unassigned }
    /\ B \in [ T -> [ F -> AmplitudeValues ] ] \cup { Unassigned }
    /\ sampleRate \in (TargetSampleRate - 1)..(TargetSampleRate + 1)


\* Properties

\* -- Sequence

\* Standardization must occur before STFT.
StandardizeBeforeSTFT ==
    [] (state = "time-frequency" =>
        x # Unassigned)

\* STFT must occur before PhaseSplit.
STFTBeforePhaseSplit ==
    [] (state = "phase-split" =>
        X # Unassigned)

\* PhaseSplit must occur before BaseLine.
PhaseSplitBeforeBaseline ==
    [] (state = "baseline" =>
        /\ M # Unassigned
        /\ PHI # Unassigned)

\* Baseline must occur before Denoise.
BaselineBeforeDenoise ==
    [] (state = "denoising" =>
        B # Unassigned)

\* -- Existence

\* A standardized signal must exist after standardization.
StandardizedSignalExists ==
    [] (state \in {"standardized",
                   "time-frequency",
                   "phase-split",
                   "baseline",
                   "denoising"} =>
        x # Unassigned)

\* Time-frequency representation must exist after STFT.
TimeFrequencyExists ==
    [] (state \in {"time-frequency",
                   "phase-split",
                   "baseline",
                   "denoising"} =>
        X # Unassigned)

\* Magnitude and phase must exist after PhaseSplit.
MagnitudePhaseExist ==
    [] (state \in {"phase-split",
                   "baseline",
                   "denoising"} =>
        /\ M # Unassigned
        /\ PHI # Unassigned)

\* Baseline output must exist before denoising.
BaselineExists ==
    [] (state = "denoising" => B # Unassigned)

\* -- High-level correctness

\* Standardization must set the target sample rate.
TargetSampleRateEnforced ==
    [] (state \in {"standardized",
                   "time-frequency",
                   "phase-split",
                   "baseline",
                   "denoising"} =>
        sampleRate = TargetSampleRate)

\* Standardization must preserve the original signal.
\* (Applicable to your current abstraction, where resampling
\*  is represented as a direct copy.)
SignalPreserved ==
    [] (state = "standardized" => x = x_0)

\* Magnitude and phase must have matching time-frequency domains.
MagnitudePhaseDomains ==
    [] (M # Unassigned /\ PHI # Unassigned =>
        DOMAIN M = DOMAIN PHI)

================================================