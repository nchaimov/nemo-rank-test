#!/bin/bash 
module load nemo
export PYTHONPATH="$PWD/site-packages:$PYTHONPATH"
mpirun -np 2 -hostfile hosts python rank.py trainer.num_nodes=2
