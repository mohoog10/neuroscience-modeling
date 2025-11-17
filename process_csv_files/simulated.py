import numpy as np
import pandas as pd
import os

def pad_dataframe(df: pd.DataFrame, max_len: int, padding_value: float = 0.0) -> pd.DataFrame:
    """
    Pad or truncate a single DataFrame (timesteps x features) to a fixed length.
    Returns a new DataFrame with the same columns, padded/truncated to max_len.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with shape (timesteps, features).
    max_len : int
        Target number of timesteps (rows).
    padding_value : float
        Value to use for padding (default 0.0).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with shape (max_len, n_features).
    """
    n_features = df.shape[1]
    length = min(df.shape[0], max_len)

    # Create padded array
    out = np.full((max_len, n_features), padding_value, dtype=np.float32)

    # Copy existing values (truncate if longer)
    out[:length, :] = df.iloc[:length, :].to_numpy(dtype=np.float32)

    # Return DataFrame with same column names
    return pd.DataFrame(out, columns=df.columns)


def create_simulated_dataset(num_samples: int, timestep_max: int, num_features: int,
                             num_targets: int, out_dir: str = "simulated_dataset"):
    """
    Create a simulated dataset with random values and target columns.
    Saves one CSV file per sample.

    Args:
        num_samples (int): Number of samples.
        timestep_max (int): Maximum number of timesteps.
        num_features (int): Number of features.
        num_targets (int): Number of target values.
        out_dir (str): Directory to save the CSV files.

    Returns:
        list: Paths to the saved CSV files.
    """
    os.makedirs(out_dir, exist_ok=True)

    file_paths = []
    for i in range(num_samples):
        # Random timesteps per sample
        timesteps = np.random.randint(int(0.5 * timestep_max), timestep_max + 1)

        # Feature matrix: (timesteps, num_features)
        features = np.random.rand(timesteps, num_features)

        # Single target value for this sample
        target = np.random.randint(0, num_targets)

        # Add target column (same value repeated for all timesteps)
        targets_col = np.full((timesteps, 1), target)

        # Concatenate features + target
        data = np.hstack([features, targets_col])

        # Build DataFrame with feature columns + target
        cols = [f"feature_{f}" for f in range(num_features)] + ["target"]
        df = pd.DataFrame(data, columns=cols)

        # Save CSV per sample
        file_path = os.path.join(out_dir, f"sample_{i}.csv")
        df.to_csv(file_path, index=False)
        file_paths.append(file_path)

    return file_paths


if __name__ == "__main__":
    files = create_simulated_dataset(
        num_samples=150,
        timestep_max=150,
        num_features=10,
        num_targets=4,
        out_dir="simulated_dataset"
    )
    csv_files = [f for f in os.listdir("simulated_dataset") if f.endswith(".csv")]
    max_len = 0
    dfs = {}
    for file in csv_files:
        file_path = os.path.join("simulated_dataset", file)
        df = pd.read_csv(file_path)
        dfs[file] = df
        if len(df)>max_len:
            max_len = len(df)
    print(f"Max length : {max_len}")
    for file, df in dfs.items():
        downsampled_df = pad_dataframe(df,200,-99999.0)#int(max_len/20),-99999.0)
        print("Original length:", len(df))
        print("Downsampled length:", len(downsampled_df))
        #filename = os.path.basename('downsampled_'+file)

        out_path = os.path.join("simulated_dataset", file)

        downsampled_df.to_csv(out_path, index=False)

