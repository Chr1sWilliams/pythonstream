```@meta
CurrentModule = Pigeons
```

# Checkpoints

Pigeons can write a "checkpoint" periodically 
to ensure that not more than half of the work is lost in 
the event of e.g. a server failure. This is enabled as follows:

```@example example
using Pigeons
pt = pigeons(target = toy_mvn_target(100), checkpoint = true)
```

See [`write_checkpoint()`](@ref) for details of how this 
is accomplished in a way compatible to both the single-machine 
and MPI contexts. 
Each checkpoint is located in 
`results/all/[unique folder]/round=[x]/checkpoint`, 
with the latest run in `results/latest/[unique folder]/round=[x]/checkpoint`. 

Checkpoints are also useful when an MPI-distributed PT has been 
ran, and the user wants to load the full set of 
results in one interactive session. 

To load a checkpoint, create a [`PT`](@ref) struct by passing in the path 
string to the checkpoint folder, for example to re-load the latest checkpoint 
from the latest run and perform more sampling:

```@example example
pt = PT("results/latest")

# do two more rounds of sampling
pt = Pigeons.increment_n_rounds!(pt, 2)
pt = pigeons(pt)
```

This is useful when you realize you will need more CPU or machines to 
help. Continuing on the above example, we will now do one more round, but 
this time using 2 local MPI processes:

```@example example
pt = Pigeons.increment_n_rounds!(pt, 1)
result = pigeons(pt.exec_folder, ChildProcess(n_local_mpi_processes = 2))
```

The above example also demonstrates that it is not necessary to load 
the PT object into the interactive node in order to increase the number of rounds. 


## Large immutable data

If part of a target is a large immutable object, it is 
wasteful to have all the machines write it at each round. 
To avoid this, encapsulate the  large immutable object 
into an `Immutable` struct. 

For an example where this is used, see
[here](https://github.com/Julia-Tempering/Pigeons.jl/blob/58e3940d0dd607a73c1b051d2282a8500fe0ec0f/src/targets/StanLogPotential.jl#L23).