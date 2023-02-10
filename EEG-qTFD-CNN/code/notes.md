
See https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html

also https://www.kaggle.com/code/elsehow/classifying-relaxation-versus-doing-math?scriptVersionId=0
https://www.kaggle.com/code/wpncrh/classifying-tasks-using-eeg-data-w-tensorflow-nn/notebook



*what to do next:*

- what features are we feeding the classifier?
  - the whole vector?
  - how to separate the channels?
- drop files that do not have a class (as we do not have the solution for those)
- separate into train and test
- pipeline:
  - transform data into features
  - pass to model for training


## Feature Extraction

- py packages:
  - tsfresh - package to extract different features
    - https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
  - pyeeg:
    - http://pyeeg.sourceforge.net/
- https://github.com/ari-dasci/S-TSFE-DL


- methods:
  - simple statistics
  - DWT
  - https://www.sciencedirect.com/science/article/pii/S174680941300089X - bag of words
  - https://ieeexplore.ieee.org/abstract/document/6090335 
  - https://www.sciencedirect.com/science/article/pii/S0950705116301174 - Automatic signal abnormality detection using time-frequency features and machine learning: A newborn EEG seizure case study
  - https://www.worldscientific.com/doi/abs/10.1142/S0129065718500302- Time-Varying EEG Correlations Improve Automated Neonatal Seizure Detection

- deep learning: https://www.sciencedirect.com/science/article/pii/S0893608019303910 (no feature extraction)

- comparison: https://www.sciencedirect.com/science/article/pii/S1388245708001405


# new notes 26/07
- pyeeg spectral_entropy
- signal coherence? https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html#scipy.signal.coherence
  - btw what? 
- entropy?
- overlap on psd_welch?
