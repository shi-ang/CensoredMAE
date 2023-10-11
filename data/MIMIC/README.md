# MIMIC-IV Dataset Download
If you are interested in using the `MIMIC` dataset, you can access the MIMIC-IV data from the [MIMIC website](https://mimic.physionet.org/) under "Accessing MIMIC-IV v2.0", or directly access this [MIMIC-IV Version 2.0](https://physionet.org/content/mimiciv/2.0/).
1. For `MIMIC-A`, you first need to go through the ethic process, and once you have done that, you can go to the 
BigQuery and process the data using the sql script `MIMIC_IV_v2.0_SMDN.sql` in the `data/MIMIC/` folder.
And further process the data using the code in `MIMIC_IV_V2.0_preprocess.py`.
2. For `MIMIC-H`, go through the same ethic process, and then use the code in `MIMIC_IV_hosp_failure.sql` 
(also in the `data/MIMIC/` folder) to acquire the data from BigQuery. 
And further process the data using the code in `MIMIC_IV_hosp_failure.py`.