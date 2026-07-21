import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



def main_data_plotting():

    # Create Plots subfolder if it doesn't exist
    plots_folder = os.path.join(os.path.dirname(__file__), "Plots")
    os.makedirs(plots_folder, exist_ok=True)

    # Define the path to the Matlab folder
    matlab_folder = (
        r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\LHS_Data"
    )

    # List of worker files
    worker_files = ["worker1.txt", "worker2.txt", "worker3.txt", "worker4.txt"]

    # Read all worker files and combine them
    dfs = []
    for worker_file in worker_files:
        file_path = os.path.join(matlab_folder, worker_file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Combined dataframe shape: {combined_df.shape}")
    print(f"\nFirst few rows:")
    print(combined_df.head())
    print(f"\nDataframe columns: {list(combined_df.columns)}")
    print(f"\nDataframe info:")
    print(combined_df.info())

    # Check for duplicates
    print("\n" + "=" * 80)
    print("DUPLICATE ANALYSIS")
    print("=" * 80)
    total_duplicates = combined_df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates}")

    if total_duplicates > 0:
        print(f"\nDuplicate rows (showing first 10):")
        print(combined_df[combined_df.duplicated(keep=False)].head(10))
    else:
        print("No duplicate rows found.")

    # Check if CO2_out, CO_out, or H2_out are negative and set Error to 2
    print("\n" + "=" * 80)
    print("CHECKING FOR NEGATIVE OUTPUT VALUES")
    print("=" * 80)

    negative_mask = (combined_df["CO2_out"] < 0) | (combined_df["CO_out"] < 0) | (combined_df["H2_out"] < 0)
    num_negative = negative_mask.sum()
    print(f"Rows with negative outputs (CO2_out < 0 OR CO_out < 0 OR H2_out < 0): {num_negative}")

    if num_negative > 0:
        combined_df.loc[negative_mask, "Error"] = 2
        print(f"Set Error = 2 for {num_negative} rows with negative outputs")

    # Filter for Error > 0
    filtered_df = combined_df[combined_df["Error"] > 0]
    print(f"\nFiltered dataframe (Error > 0): {filtered_df.shape[0]} rows")

    # Plot Error vs v and CO2share
    print("\n" + "=" * 80)
    print("PLOTTING ERROR vs v and CO2share (Error > 0 only)")
    print("=" * 80)

    columns = ["v", "CO2share", "E_appl"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, col in enumerate(columns):
        ax = axes[idx]
        ax.hist(filtered_df[col], bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "error_analysis.png"), dpi=300, bbox_inches="tight")
    print(f"Plot saved to Plots/error_analysis.png")
    plt.show()

    # Save individual histograms
    for idx, col in enumerate(columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(filtered_df[col], bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"distribution_{col}.png"), dpi=300, bbox_inches="tight")
        print(f"Individual histogram saved to Plots/distribution_{col}.png")
        plt.close()

    # Create heatmaps for Error > 0 in a single figure
    print("\n" + "=" * 80)
    print("CREATING COMBINED HEATMAPS: Error > 0")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # v vs CO2share
    heatmap_data_1 = filtered_df.pivot_table(
        values="Error",
        index=pd.cut(filtered_df["v"], bins=20),
        columns=pd.cut(filtered_df["CO2share"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_1, cmap="YlOrRd", ax=axes[0], cbar_kws={"label": "Count"})
    axes[0].set_title("v vs CO2share (Error > 0)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("CO2share", fontsize=10)
    axes[0].set_ylabel("v", fontsize=10)

    # v vs E_appl
    heatmap_data_2 = filtered_df.pivot_table(
        values="Error",
        index=pd.cut(filtered_df["v"], bins=20),
        columns=pd.cut(filtered_df["E_appl"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_2, cmap="YlOrRd", ax=axes[1], cbar_kws={"label": "Count"})
    axes[1].set_title("v vs E_appl (Error > 0)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("E_appl", fontsize=10)
    axes[1].set_ylabel("v", fontsize=10)

    # CO2share vs E_appl
    heatmap_data_3 = filtered_df.pivot_table(
        values="Error",
        index=pd.cut(filtered_df["CO2share"], bins=20),
        columns=pd.cut(filtered_df["E_appl"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_3, cmap="YlOrRd", ax=axes[2], cbar_kws={"label": "Count"})
    axes[2].set_title("CO2share vs E_appl (Error > 0)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("E_appl", fontsize=10)
    axes[2].set_ylabel("CO2share", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "combined_error_gt_zero_heatmaps.png"), dpi=300, bbox_inches="tight")
    print(f"Combined heatmaps saved to Plots/combined_error_gt_zero_heatmaps.png")
    plt.show()

    # Save individual heatmaps for Error > 0
    heatmap_names_gt_zero = [
        "v_vs_CO2share_error_gt_zero",
        "v_vs_E_appl_error_gt_zero",
        "CO2share_vs_E_appl_error_gt_zero",
    ]
    heatmap_datas_gt_zero = [heatmap_data_1, heatmap_data_2, heatmap_data_3]

    for name, data in zip(heatmap_names_gt_zero, heatmap_datas_gt_zero):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(data, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Count"})
        ax.set_title(f"Heatmap: {name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{name}.png"), dpi=300, bbox_inches="tight")
        print(f"Individual heatmap saved to Plots/{name}.png")
        plt.close()

    # Create heatmaps for Error == 0 in a single figure
    print("\n" + "=" * 80)
    print("CREATING COMBINED HEATMAPS: Error == 0")
    print("=" * 80)

    filtered_df_zero = combined_df[combined_df["Error"] == 0]
    print(f"Filtered dataframe (Error == 0): {filtered_df_zero.shape[0]} rows")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # v vs CO2share
    heatmap_data_zero_1 = filtered_df_zero.pivot_table(
        values="Error",
        index=pd.cut(filtered_df_zero["v"], bins=20),
        columns=pd.cut(filtered_df_zero["CO2share"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_zero_1, cmap="Blues", ax=axes[0], cbar_kws={"label": "Count"})
    axes[0].set_title("v vs CO2share (Error == 0)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("CO2share", fontsize=10)
    axes[0].set_ylabel("v", fontsize=10)

    # v vs E_appl
    heatmap_data_zero_2 = filtered_df_zero.pivot_table(
        values="Error",
        index=pd.cut(filtered_df_zero["v"], bins=20),
        columns=pd.cut(filtered_df_zero["E_appl"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_zero_2, cmap="Blues", ax=axes[1], cbar_kws={"label": "Count"})
    axes[1].set_title("v vs E_appl (Error == 0)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("E_appl", fontsize=10)
    axes[1].set_ylabel("v", fontsize=10)

    # CO2share vs E_appl
    heatmap_data_zero_3 = filtered_df_zero.pivot_table(
        values="Error",
        index=pd.cut(filtered_df_zero["CO2share"], bins=20),
        columns=pd.cut(filtered_df_zero["E_appl"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_zero_3, cmap="Blues", ax=axes[2], cbar_kws={"label": "Count"})
    axes[2].set_title("CO2share vs E_appl (Error == 0)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("E_appl", fontsize=10)
    axes[2].set_ylabel("CO2share", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "combined_error_eq_zero_heatmaps.png"), dpi=300, bbox_inches="tight")
    print(f"Combined heatmaps saved to Plots/combined_error_eq_zero_heatmaps.png")
    plt.show()

    # Save individual heatmaps for Error == 0
    heatmap_names_eq_zero = [
        "v_vs_CO2share_error_eq_zero",
        "v_vs_E_appl_error_eq_zero",
        "CO2share_vs_E_appl_error_eq_zero",
    ]
    heatmap_datas_eq_zero = [heatmap_data_zero_1, heatmap_data_zero_2, heatmap_data_zero_3]

    for name, data in zip(heatmap_names_eq_zero, heatmap_datas_eq_zero):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(data, cmap="Blues", ax=ax, cbar_kws={"label": "Count"})
        ax.set_title(f"Heatmap: {name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{name}.png"), dpi=300, bbox_inches="tight")
        print(f"Individual heatmap saved to Plots/{name}.png")
        plt.close()

    # Create a heatmap for all data with count as heat
    print("\n" + "=" * 80)
    print("CREATING COMBINED HEATMAPS: All data")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # v vs CO2share
    heatmap_data_all_1 = combined_df.pivot_table(
        values="Error",
        index=pd.cut(combined_df["v"], bins=20),
        columns=pd.cut(combined_df["CO2share"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_all_1, cmap="YlOrRd", ax=axes[0], cbar_kws={"label": "Count"})
    axes[0].set_title("v vs CO2share (All data)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("CO2share", fontsize=10)
    axes[0].set_ylabel("v", fontsize=10)

    # v vs E_appl
    heatmap_data_all_2 = combined_df.pivot_table(
        values="Error",
        index=pd.cut(combined_df["v"], bins=20),
        columns=pd.cut(combined_df["E_appl"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_all_2, cmap="YlOrRd", ax=axes[1], cbar_kws={"label": "Count"})
    axes[1].set_title("v vs E_appl (All data)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("E_appl", fontsize=10)
    axes[1].set_ylabel("v", fontsize=10)

    # CO2share vs E_appl
    heatmap_data_all_3 = combined_df.pivot_table(
        values="Error",
        index=pd.cut(combined_df["CO2share"], bins=20),
        columns=pd.cut(combined_df["E_appl"], bins=20),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data_all_3, cmap="YlOrRd", ax=axes[2], cbar_kws={"label": "Count"})
    axes[2].set_title("CO2share vs E_appl (All data)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("E_appl", fontsize=10)
    axes[2].set_ylabel("CO2share", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "combined_all_data_heatmaps.png"), dpi=300, bbox_inches="tight")
    print(f"Combined heatmaps saved to Plots/combined_all_data_heatmaps.png")
    plt.show()

    # Save individual heatmaps for all data
    heatmap_names_all = ["v_vs_CO2share_all_data", "v_vs_E_appl_all_data", "CO2share_vs_E_appl_all_data"]
    heatmap_datas_all = [heatmap_data_all_1, heatmap_data_all_2, heatmap_data_all_3]

    for name, data in zip(heatmap_names_all, heatmap_datas_all):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(data, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Count"})
        ax.set_title(f"Heatmap: {name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{name}.png"), dpi=300, bbox_inches="tight")
        print(f"Individual heatmap saved to Plots/{name}.png")
        plt.close()

    # Create boxplots for every variable
    print("\n" + "=" * 80)
    print("CREATING BOXPLOTS FOR ALL VARIABLES (Error == 0 only)")
    print("=" * 80)

    # Filter for Error == 0
    boxplot_df = combined_df[combined_df["Error"] == 0]
    print(f"Boxplots will use {boxplot_df.shape[0]} rows (Error == 0)")

    for col in boxplot_df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(boxplot_df[col].dropna(), vert=True)
        ax.set_ylabel(col, fontsize=11)
        ax.set_title(f"Box Plot: {col} (Error == 0)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"boxplot_{col}.png"), dpi=300, bbox_inches="tight")
        print(f"Boxplot saved to Plots/boxplot_{col}.png")
        plt.close()

    # Create 3D scatter plot
    print("\n" + "=" * 80)
    print("3D SCATTER PLOT")
    print("=" * 80)

    # Filter data where Error == 0
    filtered_df = combined_df[combined_df["Error"] == 0]
    print(f"Filtered data points (Error == 0): {len(filtered_df)}")

    # Create 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["CO2share"],
        filtered_df["E_appl"],
        filtered_df["CO2_out"],
        c=filtered_df["CO2_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("CO2 Share", fontsize=12, fontweight="bold")
    ax.set_ylabel("E_appl", fontsize=12, fontweight="bold")
    ax.set_zlabel("CO2 Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: CO2 Output vs CO2 Share and Applied Voltage\n(Error == 0)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("CO2 Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_CO2_output.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_CO2_output.png")
    plt.show()
    plt.close()

    # Create second 3D scatter plot with v (velocity) on y-axis
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["CO2share"],
        filtered_df["v"],
        filtered_df["CO2_out"],
        c=filtered_df["CO2_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("CO2 Share", fontsize=12, fontweight="bold")
    ax.set_ylabel("Velocity (v)", fontsize=12, fontweight="bold")
    ax.set_zlabel("CO2 Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: CO2 Output vs CO2 Share and Velocity\n(Error == 0)", fontsize=14, fontweight="bold", pad=20
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("CO2 Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_CO2_output_vs_v.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_CO2_output_vs_v.png")
    plt.show()
    plt.close()

    # Create third 3D scatter plot with v (velocity) on x-axis and E_appl on y-axis
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["v"],
        filtered_df["E_appl"],
        filtered_df["CO2_out"],
        c=filtered_df["CO2_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("Velocity (v)", fontsize=12, fontweight="bold")
    ax.set_ylabel("E_appl", fontsize=12, fontweight="bold")
    ax.set_zlabel("CO2 Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: CO2 Output vs Velocity and Applied Voltage\n(Error == 0)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("CO2 Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_CO2_output_vs_v_Eappl.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_CO2_output_vs_v_Eappl.png")
    plt.show()
    plt.close()

    # ============================================================================
    # Create 3D scatter plots for H2_out
    # ============================================================================
    print("\n" + "=" * 80)
    print("3D SCATTER PLOTS FOR H2_OUT")
    print("=" * 80)

    # Plot 1: H2_out vs CO2share and E_appl
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["CO2share"],
        filtered_df["E_appl"],
        filtered_df["H2_out"],
        c=filtered_df["H2_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("CO2 Share", fontsize=12, fontweight="bold")
    ax.set_ylabel("E_appl", fontsize=12, fontweight="bold")
    ax.set_zlabel("H2 Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: H2 Output vs CO2 Share and Applied Voltage\n(Error == 0)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("H2 Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_H2_output.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_H2_output.png")
    plt.show()
    plt.close()

    # Plot 2: H2_out vs CO2share and v
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["CO2share"],
        filtered_df["v"],
        filtered_df["H2_out"],
        c=filtered_df["H2_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("CO2 Share", fontsize=12, fontweight="bold")
    ax.set_ylabel("Velocity (v)", fontsize=12, fontweight="bold")
    ax.set_zlabel("H2 Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: H2 Output vs CO2 Share and Velocity\n(Error == 0)", fontsize=14, fontweight="bold", pad=20
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("H2 Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_H2_output_vs_v.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_H2_output_vs_v.png")
    plt.show()
    plt.close()

    # Plot 3: H2_out vs v and E_appl
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["v"],
        filtered_df["E_appl"],
        filtered_df["H2_out"],
        c=filtered_df["H2_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("Velocity (v)", fontsize=12, fontweight="bold")
    ax.set_ylabel("E_appl", fontsize=12, fontweight="bold")
    ax.set_zlabel("H2 Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: H2 Output vs Velocity and Applied Voltage\n(Error == 0)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("H2 Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_H2_output_vs_v_Eappl.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_H2_output_vs_v_Eappl.png")
    plt.show()
    plt.close()

    # ============================================================================
    # Create 3D scatter plots for CO_out
    # ============================================================================
    print("\n" + "=" * 80)
    print("3D SCATTER PLOTS FOR CO_OUT")
    print("=" * 80)

    # Plot 1: CO_out vs CO2share and E_appl
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["CO2share"],
        filtered_df["E_appl"],
        filtered_df["CO_out"],
        c=filtered_df["CO_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("CO2 Share", fontsize=12, fontweight="bold")
    ax.set_ylabel("E_appl", fontsize=12, fontweight="bold")
    ax.set_zlabel("CO Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: CO Output vs CO2 Share and Applied Voltage\n(Error == 0)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("CO Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_CO_output.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_CO_output.png")
    plt.show()
    plt.close()

    # Plot 2: CO_out vs CO2share and v
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["CO2share"],
        filtered_df["v"],
        filtered_df["CO_out"],
        c=filtered_df["CO_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("CO2 Share", fontsize=12, fontweight="bold")
    ax.set_ylabel("Velocity (v)", fontsize=12, fontweight="bold")
    ax.set_zlabel("CO Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: CO Output vs CO2 Share and Velocity\n(Error == 0)", fontsize=14, fontweight="bold", pad=20
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("CO Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_CO_output_vs_v.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_CO_output_vs_v.png")
    plt.show()
    plt.close()

    # Plot 3: CO_out vs v and E_appl
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        filtered_df["v"],
        filtered_df["E_appl"],
        filtered_df["CO_out"],
        c=filtered_df["CO_out"],
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("Velocity (v)", fontsize=12, fontweight="bold")
    ax.set_ylabel("E_appl", fontsize=12, fontweight="bold")
    ax.set_zlabel("CO Output", fontsize=12, fontweight="bold")
    ax.set_title(
        "3D Scatter Plot: CO Output vs Velocity and Applied Voltage\n(Error == 0)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("CO Output", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "3d_scatter_CO_output_vs_v_Eappl.png"), dpi=300, bbox_inches="tight")
    print(f"3D scatter plot saved to Plots/3d_scatter_CO_output_vs_v_Eappl.png")
    plt.show()
    plt.close()

    # ============================================================================
    # Plot FE vs CD
    # ============================================================================
    print("\n" + "=" * 80)
    print("FE vs CD PLOT")
    print("=" * 80)

    # Filter for Error == 0
    fe_cd_data = combined_df[combined_df["Error"] == 0]

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        fe_cd_data["CD"],
        fe_cd_data["FE"],
        c=fe_cd_data["CO2_out"],
        cmap="viridis",
        s=50,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel("CD", fontsize=12, fontweight="bold")
    ax.set_ylabel("FE", fontsize=12, fontweight="bold")
    ax.set_title("Faradaic Efficiency (FE) vs Current Density (CD)\n(Error == 0)", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CO2 Output", fontsize=11, fontweight="bold")

    grid = ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "FE_vs_CD.png"), dpi=300, bbox_inches="tight")
    print(f"Plot saved to Plots/FE_vs_CD.png")
    plt.show()
    plt.close()

    print("\nAnalysis complete!")


def senstitivity_data_plot_pair(folderpaths, x, y, y_function, x_name, y_name):
    """Generates two plots into adjecent folders from sensitivity data, 1 normal plot 1 with additional curvefit

    Args:
        folderpaths (list[os.path]): provides the two folderpaths for the plots, first one without curve second with
        x (pandas.series[np.float64]): x direction for plot
        y (pandas.series[np.float64]): y direction for plot
        y_function (pandas.series[np.float64] | None): y direction for plot of the curve fit bases on x direction
        x_name (str): name of x variable for plot
        y_name (str): name of y variable for plot
    """

    print("Plotting", x_name + " vs " + y_name)
    safe_name = (x_name + " vs " + y_name).replace("/", "per")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        x,
        y,
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        label="Bagemihl et al.",
    )

    ax.set_xlabel(x_name, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_name, fontsize=12, fontweight="bold")
    ax.set_title("Sensitivity Analysis: " + x_name + " vs " + y_name, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(folderpaths[0], safe_name + ".png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to", path)

    if y_function is not None:
        ax.plot(
            x,
            y_function,
            color="red",
            linewidth=2,
            label="Surrogate Model",
        )
        ax.legend(fontsize=11)
        path = os.path.join(folderpaths[1], safe_name + ".png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to", path)

    plt.close()


def sensitivity_data_plotting(data_path, folder_name):
    """
    Import sensitivity data into 3 distinct datasets and create plots:
    - sensitivity_CO2share: Sensitivity analysis on CO2share parameter
    - sensitivity_E_appl: Sensitivity analysis on E_appl parameter
    - sensitivity_v: Sensitivity analysis on v parameter
    """

    # Create Plots subfolder if it doesn't exist
    plots_folder = os.path.join(os.path.dirname(__file__), "Plots", folder_name)
    with_curve_plots_folder = os.path.join(plots_folder, "With_curve")
    without_curve_plots_folder = os.path.join(plots_folder, "Without_curve")

    try:
        os.makedirs(with_curve_plots_folder, exist_ok=True)
    except FileExistsError:
        pass

    try:
        os.makedirs(without_curve_plots_folder, exist_ok=True)
    except FileExistsError:
        pass

    def cube(x):
        return x**3

    def sqrt(x):
        return x**0.5

    # equations for plotting
    def function_CO2_out(CO2share, E_appl, v):
        return (cube(E_appl * (((sqrt(sqrt(v)) + E_appl) * -3.7203176) - 1.3941393)) + 40.244705) * CO2share

    def function_CO_out(CO2share, E_appl, v):
        return CO2share * (
            (((1.4358697 / (E_appl - (v**-0.46066585))) - ((E_appl * -0.6278232) ** 21.189863)) + (E_appl * -1.1215197))
            ** 10.426614
        )

    def function_H2_out(CO2share, E_appl, v):
        return (((E_appl * (E_appl * 0.26091927)) + 0.35308436) ** 14.994374) / v

    # Load the three sensitivity datasets
    sensitivity_CO2share = pd.read_csv(os.path.join(data_path, "sensitivity_CO2share.txt"))
    sensitivity_E_appl = pd.read_csv(os.path.join(data_path, "sensitivity_E_appl.txt"))
    sensitivity_v = pd.read_csv(os.path.join(data_path, "sensitivity_v.txt"))

    sensitivity_CO2share = sensitivity_CO2share[sensitivity_CO2share["Error"] == 0]
    sensitivity_E_appl = sensitivity_E_appl[sensitivity_E_appl["Error"] == 0]
    sensitivity_v = sensitivity_v[sensitivity_v["Error"] == 0]

    sensitivity_CO2share["function_CO2_out"] = function_CO2_out(
        sensitivity_CO2share["CO2share"], sensitivity_CO2share["E_appl"], sensitivity_CO2share["v"]
    )
    sensitivity_E_appl["function_CO2_out"] = function_CO2_out(
        sensitivity_E_appl["CO2share"], sensitivity_E_appl["E_appl"], sensitivity_E_appl["v"]
    )
    sensitivity_v["function_CO2_out"] = function_CO2_out(
        sensitivity_v["CO2share"], sensitivity_v["E_appl"], sensitivity_v["v"]
    )
    sensitivity_CO2share["function_H2_out"] = function_H2_out(
        sensitivity_CO2share["CO2share"], sensitivity_CO2share["E_appl"], sensitivity_CO2share["v"]
    )
    sensitivity_E_appl["function_H2_out"] = function_H2_out(
        sensitivity_E_appl["CO2share"], sensitivity_E_appl["E_appl"], sensitivity_E_appl["v"]
    )
    sensitivity_v["function_H2_out"] = function_H2_out(
        sensitivity_v["CO2share"], sensitivity_v["E_appl"], sensitivity_v["v"]
    )
    sensitivity_CO2share["function_CO_out"] = function_CO_out(
        sensitivity_CO2share["CO2share"], sensitivity_CO2share["E_appl"], sensitivity_CO2share["v"]
    )
    sensitivity_E_appl["function_CO_out"] = function_CO_out(
        sensitivity_E_appl["CO2share"], sensitivity_E_appl["E_appl"], sensitivity_E_appl["v"]
    )
    sensitivity_v["function_CO_out"] = function_CO_out(
        sensitivity_v["CO2share"], sensitivity_v["E_appl"], sensitivity_v["v"]
    )

    print("\n" + "=" * 80)
    print("SENSITIVITY DATA PLOTTING")
    print("=" * 80)

    folderpaths = [without_curve_plots_folder, with_curve_plots_folder]
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_CO2share["CO2share"],
        sensitivity_CO2share["CO2_out"],
        sensitivity_CO2share["function_CO2_out"],
        "CO2 Share [-]",
        "CO2 Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_CO2share["CO2share"],
        sensitivity_CO2share["H2_out"],
        sensitivity_CO2share["function_H2_out"],
        "CO2 Share [-]",
        "H2 Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_CO2share["CO2share"],
        sensitivity_CO2share["CO_out"],
        sensitivity_CO2share["function_CO_out"],
        "CO2 Share [-]",
        "CO Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["E_appl"],
        sensitivity_E_appl["CO2_out"],
        sensitivity_E_appl["function_CO2_out"],
        "E applied [V]",
        "CO2 Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["E_appl"],
        sensitivity_E_appl["H2_out"],
        sensitivity_E_appl["function_H2_out"],
        "E applied [V]",
        "H2 Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["E_appl"],
        sensitivity_E_appl["CO_out"],
        sensitivity_E_appl["function_CO_out"],
        "E applied [V]",
        "CO Concentration [mol/m³]",
    )
    v_flow = sensitivity_v["v"] * 600  # convert m/s to ml/min (A = 1e-5 m²)

    senstitivity_data_plot_pair(
        folderpaths,
        v_flow,
        sensitivity_v["CO2_out"],
        sensitivity_v["function_CO2_out"],
        "Gas Flow Rate [ml/min]",
        "CO2 Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        v_flow,
        sensitivity_v["H2_out"],
        sensitivity_v["function_H2_out"],
        "Gas Flow Rate [ml/min]",
        "H2 Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        v_flow,
        sensitivity_v["CO_out"],
        sensitivity_v["function_CO_out"],
        "Gas Flow Rate [ml/min]",
        "CO Concentration [mol/m³]",
    )

    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_CO2share["CO2share"],
        sensitivity_CO2share["CD"],
        None,
        "CO2 Share [-]",
        "Current Density [A/m²]",
    )

    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_CO2share["CO2share"],
        sensitivity_CO2share["Vcell"],
        None,
        "CO2 Share [-]",
        "Cell Voltage [V]",
    )

    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["E_appl"],
        sensitivity_E_appl["CD"],
        None,
        "E applied [V]",
        "Current Density [A/m²]",
    )

    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["E_appl"],
        sensitivity_E_appl["Vcell"],
        None,
        "E applied [V]",
        "Cell Voltage [V]",
    )

    # Mirror plots: Vcell as x-axis instead of E_appl
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["Vcell"],
        sensitivity_E_appl["CO2_out"],
        None,
        "Cell Voltage [V]",
        "CO2 Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["Vcell"],
        sensitivity_E_appl["H2_out"],
        None,
        "Cell Voltage [V]",
        "H2 Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["Vcell"],
        sensitivity_E_appl["CO_out"],
        None,
        "Cell Voltage [V]",
        "CO Concentration [mol/m³]",
    )
    senstitivity_data_plot_pair(
        folderpaths,
        sensitivity_E_appl["Vcell"],
        sensitivity_E_appl["CD"],
        None,
        "Cell Voltage [V]",
        "Current Density [A/m²]",
    )

    senstitivity_data_plot_pair(
        folderpaths,
        v_flow,
        sensitivity_v["CD"],
        None,
        "Gas Flow Rate [ml/min]",
        "Current Density [A/m²]",
    )

    senstitivity_data_plot_pair(
        folderpaths,
        v_flow,
        sensitivity_v["Vcell"],
        None,
        "Gas Flow Rate [ml/min]",
        "Cell Voltage [V]",
    )


path1 = r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\SensitivityData_095_06"
sensitivity_data_plotting(path1, "E_appl_0.95_CO2share_0.6")

path2 = r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\SensitivityData_125_08"
sensitivity_data_plotting(path2, "E_appl_1.25_CO2share_0.8")

path3 = r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\SensitivityData_145_08"
sensitivity_data_plotting(path3, "E_appl_145_CO2share_0.8")
