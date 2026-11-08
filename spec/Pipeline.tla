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

EXTENDS Reals, FiniteSets

CONSTANTS
    StateSet

VARIABLES
    state

vars == << state >>

Init == state \in StateSet

Next == state' \in StateSet

Spec == Init /\ [][Next]_vars

TypeOk ==
    /\ state \in StateSet

================================================