# NeMo Rank Test

This is a simple NeMo test which initializes NeMo by running a Trainer that does nothing and prints the rank according to PyTorch and NeMo. This demonstrates how, when launched through MVAPICH's mpirun, NeMo does not pick up the MPI rank, whereas PyTorch does. This can be fixed by setting the environment variable `$OMPI_COMM_WORLD_RANK` to the MPI rank, which can be obtained from `$PMI_RANK` as set by MVAPICH.

Running `run-2node-mpi-without-env-var.sh` on Illyad (which runs two ranks, one on Illyad and one on Gilgamesh) will print at the end of execution

```
[hostname=illyad, nemo_rank=0, torch_rank=0] Hello world!
[hostname=illyad] Success: NeMo and PyTorch agree on rank number
[hostname=gilgamesh, nemo_rank=0, torch_rank=1] Hello world!
[hostname=gilgamesh] Error: NeMo and PyTorch don't agree on rank number
```

PyTorch has the correct ranks, whereas NeMo incorrectly gets rank 0 for every rank because it it hard-coded to look for only Slurm or Open MPI environment variables.

Running `run-2node-mpi-with-env-var.sh` uses the `launch_nemo` script to set `OMPI_COMM_WORLD_RANK=$PMI_RANK`. Now the output is:

```
[hostname=illyad, nemo_rank=0, torch_rank=0] Hello world!
[hostname=illyad] Success: NeMo and PyTorch agree on rank number
[hostname=gilgamesh, nemo_rank=1, torch_rank=1] Hello world!
[hostname=gilgamesh] Success: NeMo and PyTorch agree on rank number
```

NeMo and PyTorch now agree on rank number.
