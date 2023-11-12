import numpy as np
import argparse
import glob
import re
import os

def load_samples_into_memmap(output_directory):

    replica_data_path = os.path.join(output_directory, 'replica_data')
    index_process_path = os.path.join(output_directory, 'index_process')

    max_round = 0
    max_replica = 0

    search_string = f"{replica_data_path}/*replica*_rnd*.npy"

    # Compile a pattern that matches any string ending with 'replica<number>_rnd<number>.npy'
    pattern = re.compile(r"replica(\d+)_rnd(\d+).npy")

    # Find the maximum round and replica numbers
    for file_path in glob.glob(search_string):
        match = pattern.search(file_path)
        if match:
            replica_number = int(match.group(1))
            round_number = int(match.group(2))
            max_round = max(max_round, round_number)
            max_replica = max(max_replica, replica_number)


    max_samples_per_chain = 2**max_round

    if max_round > 0 and max_replica > 0:
        # Create a directory for the memmap files if it doesn't exist
        memmap_dir = os.path.join(output_directory, f'samples')
        os.makedirs(memmap_dir, exist_ok=True)

        # Keep track of the current index for each memmap
        memmap_indices = {replica: 0 for replica in range(1, max_replica + 1)}

        first_file = True

        # Only process files from the maximum round
        max_round_path = os.path.join(replica_data_path, f'*_replica*_rnd{max_round}.npy')
        for file_path in glob.glob(max_round_path):
            match = pattern.search(file_path)
            if match:
                if first_file:

                    memmap_array_len = len(np.memmap(file_path, dtype='float32', mode='r'))

                    dim = memmap_array_len//max_samples_per_chain

                            # Initialize memmap arrays for each chain
                    chain_memmaps = {
                        replica: np.memmap(
                            os.path.join(memmap_dir, f'chain_samples_{replica}_nsample{max_samples_per_chain}_dim{dim}.dat'),
                            dtype='float32',
                            mode='w+',
                            shape=(max_samples_per_chain,dim)
                        ) for replica in range(1, max_replica + 1)
                    }

                    first_file = False

                replica_number = int(match.group(1))

                # Use memory-mapped file to avoid loading entire array into memory
                memmap_array = np.memmap(file_path, dtype='float32', mode='r', shape=(max_samples_per_chain, dim))

                # Load index data for current replica
                index_data = np.load(os.path.join(index_process_path, f"index_process_{replica_number}.npy"))

                # Process and write samples to the corresponding memmap
                for i, index in enumerate(index_data):
                    if index in chain_memmaps:
                        chain_memmaps[index][memmap_indices[index]] = memmap_array[i,:]
                        memmap_indices[index] += 1
    else:
        print("No matching files found or max_round and max_replica numbers are not valid.")
        return None

    # Flush the memmaps to ensure all data is written to disk
    for memmap in chain_memmaps.values():
        memmap.flush()

    return chain_memmaps

def main():
    parser = argparse.ArgumentParser(description="Save Samples")
    parser.add_argument("--output-dir", help="Directory to save output samples", type=str, default="/home/meta-flux/output_samples")
    #parser.add_argument("--max-round", help="Max number of rounds to save to disk", type=int, default=10)
    args = parser.parse_args()

    chain_memmaps = load_samples_into_memmap(args.output_dir)
    if chain_memmaps:
        print("Samples are written to memmap arrays.")



if __name__ == "__main__":
    main()
