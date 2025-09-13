# Data directory

This directory is the logical home for behavioural datasets used by the project.  **No raw data is included in this repository.**  You are expected to obtain datasets separately and place them here in an appropriate layout.

## Supported datasets

At the time of writing, the following public datasets are recommended for experimentation:

* **H‑MOG** – smartphone sensor and touch data from 100 users.  See [Feng et al., “Continuous Mobile Authentication using Temporal Features”].
* **BehavePassDB** – behavioural biometrics dataset including free‑text typing and swipe gestures from 81 users.  See [Stragapede et al., “BehavePassDB: A Multi‑Modal Mobile Database for Behavioral Biometrics”].
* **BrainRun** – inertial and gesture data collected from thousands of players of a mobile game.  See [Mayer et al., “Implicit Authentication via BrainRun Dataset”].

For each dataset you choose to work with, create a subdirectory named after the dataset (e.g. `hmog/`) and store the raw files there.  Your preprocessing scripts should read from these directories and output feature matrices and labels in a format consumable by the training scripts.

## Synthetic data

When experimenting with model architectures, you may wish to generate synthetic data rather than dealing with raw sensors.  The training scripts in this repository can optionally generate random feature vectors for a set of dummy users to demonstrate pipeline operation.

Synthetic samples have the form `(user_id, feature_vector, label)` where:

* `user_id` uniquely identifies a user; in synthetic mode these are integers starting from 0.
* `feature_vector` is a one‑dimensional NumPy array of fixed length containing numerical features (e.g. statistical summaries of keystroke timing or sensor readings).
* `label` indicates whether the sample belongs to the claimed user (`True` for genuine, `False` for impostor).  In the synthetic baseline the features for impostor samples are drawn from a different distribution than the genuine ones.

You can modify the synthetic data generator to approximate the statistical properties of your target application.