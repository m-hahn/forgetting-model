#!/bin/sh
for i in {1..100}
do
   ~/python-py37-mhahn autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning_SuperLong_Both_Saving.py >> ~/scr/reinforce-logs/slurm/TUNE_autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning_SuperLong_Both_Saving.py.txt
done


