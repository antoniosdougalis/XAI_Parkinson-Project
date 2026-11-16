# Multi-Head Attention-Transformer Architecture for EEG Feature classification in Parkinson's Disease

A Python-based Pre-Processing and Deep-Learning pipeline of scripts for the submitted manuscript 

Antonios. G. Dougalis. 2026, Beyond Classification Accuracy: A Physiologically Informed Framework for Interpretable EEG 
Biomarkers in Parkinson's Disease

The files used to perform the Deep Learning Classification on the EEG extracted features are freely available for public use:

1. jpersMed_myReducedWeights_Transformer.py
-Multi-Head Attention-Transformer Architecture for EEG Feature classification 

2. projParkin_Subj_DL_PARTICIPANT_ChanANDFeat_ImportanceTest.py
-Main Script for the implementation of the Transformer Deep Learning classification on EEG Features using a Leave-One-Subject-Out (LOSO) procedure. 
The Script includes the main analytic calculations & visualisation of the classification procedure implemented in the manuscript.

3. projParkin_import_AND_processData.py
-Main script that imports raw data and performs the Preprocessing pipeline implemented for this manuscript. Following this procedure the data are subjected to Analysis for feature extraction
Script executes
a. Data importation
b. Implementation of Common Average Reference (CAR)
c. Data Filtering
d. Implementation of Independent Component Analysis for Artifact removal
e. Data packing and saving for further analysis

4. projParkin_PreProcANDFeatExtract_Util_v1.py
-Main script with functions to perform several procedures of feature extactions from the data set, including
a. time domain statsitics: compute_epoch_stats
b. Welch PSD decomposition: compute_psd_welch
c. aperiodic and periodic spectral decomposition via FOOOF toolbox: aperiodic_periodic_Spectral
d. irasa algorithm for aperiodic/periodic spectral estimation via neurodsp toolbox: compute_irasa_aperiodic
e. PLI, PLV and wPLI conncetvitiy via MNE toolbox: compute_mne_connectivity
f. create and utlise a complex Morlet Wavelet Family for spectrasl decompositions (filtering, phase extraction and PSD):createComplexWaveletFamily,
   fftWavelet, compute_Morlet_Spectrum
g. Compute modulation index for phase coupling based on the method of Tort et al., 2012 J Neurophysiol: compute_pac, pacMI
h. Compute phase-based connectivity: compute_phase_timeSeries, compute_plv_pli, compute_plv_pli_wpli
i. PyBispectral PAC and PPC via Bicoherence Toolbox: compute_Bicoherence_PAC, compute_Bicoherence_PPC, compute_timeDelayEstimates
j. compute Entropy and  mutual Information connectivity: compute_entropy, compute_mutInfo
k. Compute Lempel-Ziv temporal complexity score: compute_Lempel_Ziv_Score
l. Frequency Sliding procedure according to Cojen et al., 2014 J Neurosci: make_bandPass_filter, compute_FreqSliding, compute_masked_regions
m. Harmonic Lock of Frequency Sliding: compute_FreqSlide_HarmonicRatioLock


please contact me for any issues with the code

Dr. Antonios G. Dougalis  
Email: antoniosdougalis@med.uoc.gr; antoniosdougalis@gmail.com  
ORCID: https://orcid.org/0000-0002-2139-1616
