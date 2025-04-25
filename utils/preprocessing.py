import pandas as pd 

def preprocess_CelebA_labels(input_path:str, output_path:str) -> None: 
    """
    Changes the CelebA labels to a one-hot encoding instead of using 1 and -1. 

    Args: 
        input_path: The labels csv to read in.
        output_path: Create a new csv into this path. 
    """

    label_df = pd.read_csv(input_path)
    label_df.iloc[:, 1:] = (label_df.iloc[:, 1:] + 1)//2 # converts to one-hot 
    label_df.to_csv(output_path, index=False)
    print(f"Processed labels saved into {output_path}.")
