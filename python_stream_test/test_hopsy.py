import hopsy
import numpy as np
import argparse
import random
import os

def sample_init():
    return np.array([0.0,0.0,0.0], dtype=np.float32)

class LinearPathModel:
    def __init__(self, proposal_model: hopsy.Gaussian, target_model: hopsy.Gaussian, beta: float):
        self.proposal_model = proposal_model
        self.target_model = target_model
        self.beta = beta

    def compute_negative_log_likelihood(self, x: np.ndarray) -> float:
        return (1-self.beta)*self.proposal_model.compute_negative_log_likelihood(x) + self.beta*self.target_model.compute_negative_log_likelihood(x)

# def sample_problem(x, problem: hopsy.Problem, rng: hopsy.RandomNumberGenerator):

# 	chain = hopsy.MarkovChain(problem, hopsy.GaussianProposal)
# 	chain.proposal.stepsize = 0.1
# 	chain.proposal.state = x 
# 	accrate, states = hopsy.sample(chain, rng, n_samples=50)

# 	return chain.state

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
	parser = argparse.ArgumentParser(description="Hopsy Sampler")
	parser.add_argument("--seed", help="Random seed for number generator", type=int, required=True)
	parser.add_argument("--replica", help="Replica number", type=int, required=True)
	parser.add_argument("--output-dir", help="Directory to save output samples", type=str, default="/home/meta-flux/output_samples")
	parser.add_argument("--batch-size", help="Number of samples to save at once", type=int, default=1024)
	parser.add_argument("--max-round", help="Max number of rounds to save to disk", type=int, default=10)
	args = parser.parse_args()

	seed = np.uint32(args.seed)
	rng = hopsy.RandomNumberGenerator(seed)

	proposal_model = hopsy.Gaussian(mean=[0, 0, 0])
	target_model = hopsy.Gaussian(mean=[0.8, 0.6, 0.6],covariance=[[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01] ])
	path_model = LinearPathModel(proposal_model,target_model,0)

	A, b = [[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0, 0]

	x = sample_init()

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
			path_model.beta = beta 
			print(f"response({-path_model.compute_negative_log_likelihood(x)})")
		elif command.startswith("call_sampler!"):
			beta = float(command.split("(")[1].split(")")[0])
			rnd = int(command.split("[")[1].split("]")[0])
			scan = int(command.split("{")[1].split("}")[0])

			path_model.beta = beta

			if beta == 0:
				x = sample_init()
			else:
				problem = hopsy.Problem(A,b,path_model)
				problem.starting_point = x
				chain = hopsy.MarkovChain(problem, hopsy.GaussianProposal)
				chain.proposal.stepsize = 0.1
				accrate, states = hopsy.sample(chain, rng, n_samples=1)
				x = chain.state

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



if __name__ == "__main__":
	main()	

