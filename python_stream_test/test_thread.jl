using Pkg
Pkg.activate("Pigeons.jl") ##


# using MPI
using Pigeons  # Assuming Pigeons is compatible with MPI
using MPI


struct IteratorsContext
    round::Int
    scan::Int
end

struct SharedContext
    iterators::IteratorsContext
end


MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# Each process will execute this
if rank < 5  # Assuming you want to use 2 processes
    shared = SharedContext(IteratorsContext(1, 1))
    python_cmd = Cmd(`python3 python_examples/import_error_example.py --seed $rank --replica $rank`)
    streamstate = Pigeons.PythonStreamState(python_cmd, rank)
    beta_ps = Pigeons.PythonStreamPotential(0.1)
    for _ in 1:20

        Pigeons.call_sampler!(beta_ps, streamstate, shared)

    end
    result = Pigeons.invoke_worker(streamstate, "log_potential(0.1)", Float64)

    # Process-specific output
    println("Result from process $rank: $result")
end

MPI.Finalize()
