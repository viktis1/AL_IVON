#!/bin/sh 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J random_sampler
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- Specify we only want 1 host machine
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=10GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 11GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 12:00
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo random_sampler.out 
#BSUB -eo random_sampler.err 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s203957@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 

module load python3/3.10.13
source venv_1/bin/activate

python random_sampler.py


