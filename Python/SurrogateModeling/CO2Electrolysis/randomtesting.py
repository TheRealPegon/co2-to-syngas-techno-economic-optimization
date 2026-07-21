import os
import pandas as pd
from Python.SurrogateModeling.CO2Electrolysis.prepare_training_data import prepare_training_data

matlab_folder = (
    r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\NewData"
)

# List of worker files
worker_files = ["worker1.txt", "worker2.txt", "worker3.txt", "worker4.txt"]

X, Y, combined_df = prepare_training_data(matlab_folder=matlab_folder)
combined_df.reset_index(drop=True, inplace=True)
print(combined_df.head())


# Print specific rows for indices 1, 29, 63
print("\n" + "=" * 80)
print("SPECIFIC ROWS: v, CO2share, E_appl, CO2_out")
print("=" * 80)

indices = [1, 29, 64, 150]
for idx in indices:
    if idx < len(combined_df):
        row = combined_df.loc[idx, ["v", "CO2share", "E_appl", "CO2_out"]]
        print(f"\nIndex {idx}:")
        print(f"  v:        {row['v']:.6f}")
        print(f"  CO2share: {row['CO2share']:.6f}")
        print(f"  E_appl:   {row['E_appl']:.6f}")
        print(f"  CO2_out:  {row['CO2_out']:.6f}")
    else:
        print(f"\nIndex {idx}: Out of range (dataframe has {len(combined_df)} rows)")

print("\n" + "=" * 80)

# Write combined_df to CSV
output_csv = os.path.join(os.path.dirname(__file__), "combined_data.csv")
combined_df.to_csv(output_csv, index=False)
print(f"\nDataframe saved to: {output_csv}")
print(f"Shape: {combined_df.shape}")
print("=" * 80)
