abstract type PythonStream end

"""
$SIGNATURES

Return [`PythonStreamState`](@ref) by following these steps:

1. create a `Cmd` that uses the provided `rng` to set the random seed properly, as well 
    as target-specific configurations provided by `target`.
2. Create [`PythonStreamState`](@ref) from the `Cmd` created in step 1 and return it.
"""
initialization(target::PythonStream, rng::SplittableRandom, replica_index::Int64) = @abstract 

""" 
States used in the replicas when a [`PythonStream`](@ref) is used. 
"""
struct PythonStreamState 
    worker_process::ExpectProc
    replica_index::Int
    """ 
    $SIGNATURES 

    Create a worker process based on the supplied `cmd`. 
    The work for the provided `replica_index` will be delegated to it.

    See [`PythonStream`](@ref).
    """ 
    function PythonStreamState(cmd::Cmd, replica_index::Int)
        worker_process = 
            ExpectProc(
                cmd,
                Inf # no timeout
            )
        return new(worker_process, replica_index)
    end
end

# Internals

struct PythonStreamPath end 

#= 
Only store beta, since the worker process
will take care of path construction
=#
@auto struct PythonStreamPotential 
    beta
end

create_state_initializer(target::PythonStream, ::Inputs) = target  
default_explorer(target::PythonStream) = target 

#= 
Delegate exploration to the worker process.
=#
function step!(explorer::PythonStream, replica, shared)
    log_potential = find_log_potential(replica, shared.tempering, shared)
    call_sampler!(log_potential, replica.state, shared)
end

#= 
Delegate iid sampling to the worker process.
Same call as explorer, rely on the worker to 
detect that the annealing parameter is zero.
=#
sample_iid!(log_potential::PythonStreamPotential, replica, shared) = 
    call_sampler!(log_potential, replica.state, shared)

create_path(target::PythonStream, ::Inputs) = PythonStreamPath()

interpolate(path::PythonStreamPath, beta) = PythonStreamPotential(beta)

(log_potential::PythonStreamPotential)(state::PythonStreamState) = 
    invoke_worker(
            state, 
            "log_potential($(log_potential.beta))", 
            Float64
        )

call_sampler!(log_potential::PythonStreamPotential, state::PythonStreamState, shared) = 
    invoke_worker(
        state, 
        "call_sampler!($(log_potential.beta))[$(shared.iterators.round)]{$(shared.iterators.scan)}"
    )

# convert a random UInt64 to positive Int64/Java-Long by dropping the sign bit
java_seed(rng::SplittableRandom) = (rand(split(rng), UInt64) >>> 1) % Int64

#=
Simple stdin/stdout text-based protocol. 
=#
function invoke_worker(
        state::PythonStreamState, 
        request::AbstractString, 
        return_type::Type = Nothing)

    println(state.worker_process, request)
    prefix = expect!(state.worker_process, "response(")
    if state.replica_index == 1 && length(prefix) > 3
        # display output for replica 1 to show e.g. info messages
        print(prefix)
    end
    response_str = expect!(state.worker_process, ")")
    return return_type == Nothing ? nothing : parse(return_type, response_str)
end
