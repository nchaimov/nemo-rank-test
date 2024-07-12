#!/bin/bash 
module load nemo
mpirun -np 2 -hostfile hosts python rank.py trainer.num_nodes=2
