struct PythonTarget <: PythonStream
    command::Cmd
end

initialization(target::PythonTarget, rng::SplittableRandom, replica_index::Int64) = 
    PythonStreamState(
        `$(target.command) --seed $(split(rng).seed) --replica $(replica_index)`,
        replica_index)