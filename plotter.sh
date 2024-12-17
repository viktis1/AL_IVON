#!/bin/sh 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J plotter
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- Specify we only want 1 host machine
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=3GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 4GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 0:10
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo plotter.out 
#BSUB -eo plotter.err 

module load python3/3.10.13
source venv_1/bin/activate

python plotter.py


