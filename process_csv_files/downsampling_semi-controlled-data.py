import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
TARGET_DIR = Path("Interpolated")
OUTPUT_DIR = Path("Downsampled")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def remove_long_zero_gaps(df, spike_col="spike", max_gap=500, context=10)-> pd.DataFrame: 
    """
    Remove long stretches of zeros between spikes.
    Keeps spikes and a small context window around them.
    """
    spike_indices = df.index[df[spike_col] == 1].to_numpy()
    keep_indices = set()

    for i in range(len(spike_indices)):
        idx = spike_indices[i]
        # Always keep the spike itself
        keep_indices.add(idx)

        # Add context window around spike
        keep_indices.update(range(max(0, idx-context), min(len(df), idx+context+1)))

        # If not the last spike, check gap to next
        if i < len(spike_indices)-1:
            gap = spike_indices[i+1] - idx
            if gap <= max_gap:
                # Keep everything in the gap if it's small
                keep_indices.update(range(idx, spike_indices[i+1]+1))
            # else: skip the zeros in this gap

    return df.loc[sorted(keep_indices)]

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




if __name__ == '__main__':

    csv_files = [f for f in os.listdir(TARGET_DIR) if f.endswith(".csv")]
    max_len = 0;
    dfs = {}
    for file in csv_files:
        file_path = os.path.join(TARGET_DIR, file)
        df = pd.read_csv(file_path)
        dfs[file] = df
        if len(df)>max_len:
            max_len = len(df)
    print(f"Max length : {max_len}")
    for file, df in dfs.items():
        downsampled_df = remove_long_zero_gaps(df, spike_col="Nerve_spike", max_gap=200, context=10)
        print(f"Intermediate length: {len(downsampled_df)}")
        downsampled_df = pad_dataframe(downsampled_df,1000,-99999.0)#int(max_len/20),-99999.0)
        print("Original length:", len(df))
        print("Downsampled length:", len(downsampled_df))
        filename = os.path.basename('downsampled_'+file)

        out_path = os.path.join(OUTPUT_DIR, filename)

        downsampled_df.to_csv(out_path, index=False)
        print(f"Saved {filename} -> {out_path}")