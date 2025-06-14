import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import random

# Load data
df = pd.read_csv('signal_data1.csv')

# Convert seconds to datetime
base_time = pd.Timestamp("2023-01-01 00:00:00")
df['Time'] = df['Time'].apply(lambda x: base_time + timedelta(seconds=x))
df = df.sort_values('Time').reset_index(drop=True)

# Columns to simulate
data_cols = df.columns[1:]
original_data = df[data_cols].copy().reset_index(drop=True)

# Parameters
n_hours_needed = 7300
window_min, window_max = 12, 24  # Sample random windows of 12–24 hours
generated_data = []

while len(generated_data) < n_hours_needed:
    win_len = random.randint(window_min, window_max)
    start_idx = random.randint(0, len(original_data) - win_len)
    chunk = original_data.iloc[start_idx:start_idx + win_len].copy()
    
    # Add realistic industrial noise
    for col in data_cols:
        std = original_data[col].std()
        noise = np.random.normal(0, 0.03 * std, size=len(chunk))  # 3% std noise
        chunk[col] += noise

    generated_data.append(chunk)

# Concatenate and trim to 7300 hours
generated_df = pd.concat(generated_data, ignore_index=True).iloc[:n_hours_needed]
generated_df['Time'] = [base_time + timedelta(hours=i) for i in range(n_hours_needed)]

# Save to CSV
generated_df.to_csv("simulated_signal_data.csv", index=False)
print("Simulated data saved to 'simulated_signal_data.csv'")

# Plot comparison
plt.figure(figsize=(14, 8))
for col in data_cols:
    plt.plot(df['Time'], df[col], label=f'Original {col}', linewidth=2)
    plt.plot(generated_df['Time'], generated_df[col], '--', label=f'Simulated {col}', alpha=0.6)

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Original vs Simulated Industrial-Like Time Series")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


