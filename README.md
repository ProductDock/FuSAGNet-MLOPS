# FuSAGNet — Refactored for Governed MLOps

This repository is a refactored and modernized version of the original [FuSAGNet](https://github.com/seansihohan/FuSAGNet) framework. This refactor is based on the original repository at Commit SHA: 9ab537e.
It has been specifically adapted to support **Formal State Transitions**, **Tripple Bind** and **Resource Resolution Function** as described in:

> **"A GitOps-Driven MLOps Framework: Formalizing Reproducibility and State Transitions in the Machine Learning Lifecycle"**

---
## Attribution

This project includes code derived from FuSAGNet: [github.com/seansihohan/FuSAGNet](https://github.com/seansihohan/FuSAGNet)

Original work licensed under Apache License 2.0.

## Local Setup & Reproducibility

To recreate the experiments, follow this workflow to link the code with the versioned data stored in Google Cloud.

1. ### Authenticate with Google Cloud:

     ```bash 
          gcloud auth application-default login
 
2. ### Initialize and Configure DVC

    ```bash 
        dvc init
        dvc remote add -d mygcs gs://[YOUR_BUCKET_NAME]/[DVC_PATH]

2. ### Execute the pipeline

    ```bash 
        dvc repro

     