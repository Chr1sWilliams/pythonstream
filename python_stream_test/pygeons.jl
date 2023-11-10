#!/home/meta-flux/julia-1.9.3/bin/julia
using Pkg
Pkg.activate("Pigeons.jl") ##


using Pigeons
using NPZ

function main()
    println("Enter the path to python_sampler (Press Enter to use the default path): ")
    script_path = readline()
    if script_path == ""
        script_path = "Python_Gaussian.py"
    end

    println("Enter the number of chains (n_chains, default 12): ")
    input = readline()
    n_chains = isempty(input) ? 12 : parse(Int, input)

    println("Enter the number of rounds (n_rounds, default 10): ")
    input = readline()
    n_rounds = isempty(input) ? 10 : parse(Int, input)

    println("Enter the path to the output directory (Press Enter to use the default path): ")
    output_directory = readline()
    if output_directory == ""
        output_directory = "/home/meta-flux/output_samples/"
    end

    script_name_without_extension = splitext(basename(script_path))[1]

    # Convert round number and number of chains to strings for concatenation
    rounds_str = string(n_rounds)
    chains_str = string(n_chains)

    # Create a directory name that includes the script name, number of rounds, and number of chains
    directory_name = script_name_without_extension * "_rounds" * rounds_str * "_chains" * chains_str

    # Join this new directory name with the output directory path
    output_directory = joinpath(output_directory, directory_name)

    # Check if the directory exists, if not, create it
    if !isdir(output_directory)
        mkpath(output_directory)
    end

    replica_data_path = joinpath(output_directory, "replica_data")
    mkpath(replica_data_path)

    index_process_path = joinpath(output_directory, "index_process")
    mkpath(index_process_path)

    target = Pigeons.PythonTarget(`python3 $script_path --max-round $n_rounds --output-dir $replica_data_path `)
    pt = pigeons(target = target, n_chains = n_chains, n_rounds = n_rounds, record = [index_process,round_trip] )#,on = ChildProcess(n_local_mpi_processes = 2) )
    

    if hasproperty(pt, :reduced_recorders) && hasproperty(pt.reduced_recorders, :index_process)
        for (k, v) in pt.reduced_recorders.index_process
            npzwrite("$index_process_path/index_process_$k.npy", v)
        end
        println("Data has been saved successfully to $output_directory")
        
        # Run save_samples.py at the end of the script
        try
            run(`python3 save_samples.py --output-dir $output_directory`) # Replace with the correct path to your save_samples.py
            println("save_samples.py executed successfully.")
        catch e
            println("An error occurred while executing save_samples.py: $e")
        end
        
    else
        println("The required data structure is not present.")
    end
end

main()
