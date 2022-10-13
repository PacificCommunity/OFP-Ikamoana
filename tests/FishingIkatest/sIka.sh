#!/bin/bash
###SBATCH -t 48:00:00
#SBATCH -t 5-00:00:00
#SBATCH -N 1 
#SBATCH -n 3
###SBATCH --ntasks=30
#SBATCH -p normal
###SBATCH --mem-per-cpu=1000MB
#SBATCH --mem-per-cpu=5994
##SBATCH --mem-per-cpu=500
###SBATCH -w node04 


## singlue memory per cpu: around 200MB
## memory per core available: 191828 / 32 = 5994.625
cd /nethome/3830241/ikamoana/OFP-Ikamoana/tests/FishingIkatest 

python IkaSeapoFADym.py

wait
