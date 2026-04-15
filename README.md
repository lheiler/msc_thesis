# Evaluating Computational Brain Models as Dimensionality Reduction Methods for EEG
**Author:** Lorenz Heiler  
**Supervisors:** Dr. Pedro Mediano, Dr. Gregory Scott  
**Institution:** Imperial College London (MSc Individual Project)

## Project Overview
This thesis benchmarks four families of **Computational Brain Models (CBMs)**—cortico-thalamic, Jansen-Rit, Wong-Wang, and Hopf—as dimensionality reduction tools for clinical EEG. 

These mechanistic models are compared against traditional data-driven baselines, including PCA, spectral autoencoders, EEGNet-style autoencoders, and the catch22 feature set. 

## Key Contributions
* **Unified Benchmark:** A modular pipeline built on the **Temple University Hospital Abnormal EEG Corpus (TUH-AB)** and the **LEMON** dataset, with extensibility to other corpora.
* **Hybrid Approach:** Implementation of **amortized parameter-inference** for the cortico-thalamic model, achieving 78.4% accuracy in abnormality screening while maintaining physiological interpretability.
* **Comparative Analysis:** Evaluation of latent space quality based on dimensionality efficiency, geometry preservation, and information content.

## Repository Structure
* **/code**: Core implementation, model training, and parameter inference.
  * *See the [Detailed Code README](./code/README.md) for execution instructions.*
* **/Datasets**: EEG data and preprocessing artifacts.
* **final_report.pdf**: The completed thesis document.

## Datasets
This project utilizes:
* **TUH Abnormal EEG Corpus (v3.0.1):** 2,993 sessions of clinical EEG, used for abnormality classification.
* **LEMON (Leipzig Study for Mind-Body-Emotion Interactions):** Healthy participant resting-state EEG, used for age classification.
