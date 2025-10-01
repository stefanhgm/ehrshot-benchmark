import os
import argparse
import pandas as pd
import numpy as np
import pickle

RESULTS_FILE_NAME = 'llm_features.pkl'

def main(args):

    # Read one input source embeddings file after the other
    # llm_source_embeddings_list contains a list of file paths from 1 to n to pickle files
    llm_source_embeddings_list = args.llm_source_embeddings_list.split(' ')

    result = list(pd.read_pickle(llm_source_embeddings_list[0]))
    dimensions = result[0].shape
    print(f"Initial dimensions: {dimensions}")
    
    for i in range(1, len(llm_source_embeddings_list)):
        temp = pd.read_pickle(llm_source_embeddings_list[i])
        # Assert that position 1 to 4 of tuple are the same as in result
        for j in range(1, 4):
            assert (result[j] == temp[j]).all(), f"Position {j} of tuple is not the same in {llm_source_embeddings_list[0]} and {llm_source_embeddings_list[i]}"
        result[0] = np.concatenate((result[0], temp[0]), axis=1)
    result = tuple(result)

    assert result[0].shape[1] == dimensions[1] * len(llm_source_embeddings_list), "Dimensions do not match after concatenation"
    print(f"Final dimensions: {result[0].shape}")

    output_dir = os.path.join(args.output_dir, RESULTS_FILE_NAME)
    with open(output_dir, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved results to {output_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine embeddings of LLM and CLMBR")
    parser.add_argument("--llm_source_embeddings_list", required=True, help="LLM source embeddings")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    main(args)