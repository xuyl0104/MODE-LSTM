#!/bin/bash
#SBATCH -J paraGen3
#SBATCH -p gpucompute
# Pick nodes with feature 'foo'. Different clusters have
# different features available
# but most of the time you don't need this
# next is for specified constraints of features, p100 is for envidia
# Memory
# GPU
#SBATCH --gres=gpu:2
module load cuda10.0/toolkit/10.0.130 cudnn/7.4.2
echo "starting sentence_classification1.14"
source activate sentence_classification1.14
echo "virenv has been activated"
echo "run exeperiments"

echo "testing for sentence_classification on yelp
"
######################## data processing ###########################
# data pre-process sst5
#python text_process.py

# data pre-process yelp
#python text_process_yelp.py

######################## model training ############################
# train the model on sst5
#python main.py

# train the model on yelp
python main_yelp.py

######################## testing ###################################
#python playground.py

echo "ending"
