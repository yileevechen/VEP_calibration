## Simulation Pipeline for Gene/Cluster Variant Distribution Modeling

This workflow fits probabilistic distributions to **Pathogenic (P)** and **Benign (B)** variant scores within a given **gene** or **cluster**, identifies the **best-fitting distribution**, and generates **30 simulated datasets** using that model for downstream calibration and benchmarking.

---

###  Usage

#### **Run the universal wrapper script**
```bash
bash 00.wrapper_generate_simuData.sh <working_directory> <cluster_number> <cluster_prior> <predictor_type>
```
#### **Example**
```bash
bash 00.wrapper_generate_simuData.sh \
    cluster_simu_analysis_AM \
    9 \
    0.0441 \
    AM
```
