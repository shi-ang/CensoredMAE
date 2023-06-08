#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/shiang/Documents/Survival_Prediction/MAE4Survival/SurvivalEVAL"

data=MIMIC-IV_hosp
syn_dist=uniform

python3 main.py --dataset $data --censor_dist $syn_dist --model MTLR --lr 0.001
python3 main.py --dataset $data --censor_dist $syn_dist --model CoxPH --lr 0.01
python3 run_baseline.py --dataset $data --censor_dist $syn_dist

# RunSCA using SCA environment (python 3.6 with TensorFlow 1.8)
#python3 run_CSA.py --dataset $data --censor_dist $syn_dist

