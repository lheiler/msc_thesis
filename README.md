# Evaluating Computational Brain Models as Dimensionality Reduction Methods for EEG
**Author:** Lorenz Heiler  
**Supervisors:** Dr. Pedro Mediano, Dr. Gregory Scott  
[cite_start]**Institution:** Imperial College London (MSc Individual Project) [cite: 3, 4, 7, 9]

## Project Overview
[cite_start]This thesis benchmarks four families of **Computational Brain Models (CBMs)**—cortico-thalamic, Jansen-Rit, Wong-Wang, and Hopf—as dimensionality reduction tools for clinical EEG[cite: 19]. 

[cite_start]These mechanistic models are compared against traditional data-driven baselines, including PCA, spectral autoencoders, EEGNet-style autoencoders, and the catch22 feature set[cite: 19, 82]. 

## Key Contributions
* [cite_start]**Unified Benchmark:** A modular pipeline built on the **Temple University Hospital Abnormal EEG Corpus (TUH-AB)**[cite: 18, 81].
* [cite_start]**Hybrid Approach:** Implementation of **amortized parameter-inference** for the cortico-thalamic model, achieving 78.4% accuracy in abnormality screening while maintaining physiological interpretability[cite: 20, 27, 84].
* [cite_start]**Comparative Analysis:** Evaluation of latent space quality based on dimensionality efficiency, geometry preservation, and information content[cite: 22, 952].

## Repository Structure
* **/code**: Core implementation, model training, and parameter inference. 
  * *See the [Detailed Code README](./code/README.md) for execution instructions.*
* **/Datasets**: EEG data and preprocessing artifacts (Managed via **Git LFS**).
* **/model_comparison_results_complete**: Logs and performance metrics for all evaluated models.
* **/testing**: Validation scripts and unit tests.

## Dataset
This project utilizes the **TUH Abnormal EEG Corpus (v3.0.1)**, consisting of 2,993 sessions[cite: 18, 436]. Note that large data files are stored using Git LFS.
