import numpy as np
import argparse
import os

def log_likelihood(x: np.ndarray, beta: float) -> float:
    return -((1 - beta) * np.dot(x, x)/2 + beta * np.dot(x - 5, x - 5))

def initialize_x():
    return np.array([0.0], dtype=np.float32)

def iid_sample(dim, rng: np.random.Generator):
    return rng.normal(0, 1, dim)

def metropolis_hastings_step(current_point: np.ndarray, beta: float, rng: np.random.Generator, proposal_sd: float = 1.0) -> np.ndarray:
    dim = len(current_point)
    if beta == 0:
        return iid_sample(dim,rng)
    else:
        current_log_likelihood = log_likelihood(current_point, beta)
        proposal = current_point + rng.normal(0, proposal_sd, dim)
        proposal_log_likelihood = log_likelihood(proposal, beta)
        acceptance_ratio = np.exp(proposal_log_likelihood - current_log_likelihood)
        return proposal if rng.uniform() < acceptance_ratio else current_point


def initialize_memory_map(replica, rnd, max_samples, sample_dimension ,output_directory, dtype=np.float32):
    filename = os.path.join(output_directory, f'python_samples_replica{replica}_rnd{rnd}.npy')
    if not os.path.exists(filename):
        mmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=(max_samples, sample_dimension))
    else:
        mmap_array = np.memmap(filename, dtype=dtype, mode='r+', shape=(max_samples, sample_dimension))
    return mmap_array

def save_samples_to_mmap(mmap_array, buffer, start_index):
    mmap_array[start_index:start_index+len(buffer),:] = buffer
    mmap_array.flush()  

def main():
    parser = argparse.ArgumentParser(description="Metropolis-Hastings Sampler")
    parser.add_argument("--seed", help="Random seed for number generator", type=int, required=True)
    parser.add_argument("--replica", help="Replica number", type=int, required=True)
    parser.add_argument("--output-dir", help="Directory to save output samples", type=str, default="/home/meta-flux/output_samples")
    parser.add_argument("--batch-size", help="Number of samples to save at once", type=int, default=1024)
    parser.add_argument("--max-round", help="Max number of rounds to save to disk", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.default_rng(seed=args.seed)
    x = initialize_x()
    
    sample_dimension = len(x)  
    max_rnd = args.max_round
    max_samples = 2**max_rnd
    batch_size = min(args.batch_size,max_samples)
    output_directory = args.output_dir
    samples_buffer = np.empty((batch_size, sample_dimension), dtype=np.float32)
    buffer_index = 0 
    buffer_count = 0

    os.makedirs(output_directory, exist_ok=True)

    while True:
        command = input()
        if command.startswith("log_potential"):
            beta = float(command.split("(")[1].split(")")[0])
            print(f"response({log_likelihood(x, beta)})")
        elif command.startswith("call_sampler!"):
            beta = float(command.split("(")[1].split(")")[0])
            rnd = int(command.split("[")[1].split("]")[0])
            scan = int(command.split("{")[1].split("}")[0])
            x = metropolis_hastings_step(x, beta, rng)

            if rnd == max_rnd:
                
                samples_buffer[buffer_index] = x
                buffer_index += 1

                if buffer_index == batch_size or scan == max_samples:
                    mmap_array = initialize_memory_map(args.replica, rnd, max_samples, sample_dimension, output_directory)
                    save_samples_to_mmap(mmap_array, samples_buffer[:buffer_index], buffer_count*batch_size)
                    buffer_index = 0  
                    buffer_count += 1 
            print("response()")
        else:
            raise ValueError(f"Unknown command: {command}")

    # No need for a final save after the loop because it never ends.

if __name__ == "__main__":
    main()


