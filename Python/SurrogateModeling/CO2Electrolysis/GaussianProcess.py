import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, DotProduct
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


def prepare_training_data(
    matlab_folder: str = r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab",
    worker_files: list = ["worker1.txt", "worker2.txt", "worker3.txt", "worker4.txt"],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load and prepare training data from worker files."""

    # Read all worker files and combine them
    dfs = []
    for worker_file in worker_files:
        file_path = os.path.join(matlab_folder, worker_file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    if len(dfs) == 0:
        print("No dataframes were loaded. Check file paths.")
        return None, None

    # Combine all dataframes into one
    combined_df = pd.concat(dfs, ignore_index=True)

    original_rows = len(combined_df)

    # Drop all rows with Error > 0
    combined_df = combined_df[combined_df["Error"] == 0]
    filtered_rows = len(combined_df)
    dropped_rows = original_rows - filtered_rows

    print("\n" + "=" * 80)
    print("DATA LOADING AND FILTERING SUMMARY")
    print("=" * 80)
    print(f"Original rows:                {original_rows}")
    print(f"Rows after filtering (E==0):  {filtered_rows}")
    print(f"Rows dropped (E>0):           {dropped_rows}")
    print(f"Average time (E==0):          {combined_df['time'].mean():.4f}")

    print("\n" + "=" * 80)
    print("DATAFRAME OVERVIEW")
    print("=" * 80)
    print(f"Dataframe shape: {combined_df.shape}")
    print(f"Columns: {', '.join(combined_df.columns)}")

    X = combined_df[["v", "CO2share", "E_appl"]]
    Y = combined_df[["CO2_out", "CO_out", "H2_out", "Vcell"]]
    print(X.head(40))
    print(Y.head(40))
    return X, Y


def train_test_gaussian_process(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernel_type: str = "rbf",
    output_dir: str = None,
    random_state: int = 42,
    output_col: str = "CO2_out",
    scale_inputs: bool = True,
) -> dict:
    """
    Train and test a Gaussian Process Regressor.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training input features
    X_test : pd.DataFrame
        Test input features
    y_train : np.ndarray
        Training target values
    y_test : np.ndarray
        Test target values
    kernel_type : str
        Type of kernel: 'rbf' or 'matern'
    output_dir : str
        Directory to save results
    random_state : int
        Random seed for reproducibility
    scale_inputs : bool
        Whether to scale input data based on physical bounds

    Returns:
    --------
    dict : Dictionary containing model, predictions, and metrics
    """

    # Create output directory if specified
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Scale input data if requested
    if scale_inputs:
        print("\nScaling input data based on physical bounds:")
        scaler = MinMaxScaler()

        # Define bounds for each variable
        bounds = {
            "v": (1 / 600, 1 / 40),  # velocity bounds
            "CO2share": (0.2, 1.0),  # CO2 share bounds
            "E_appl": (-2.0, -0.5),  # E_appl bounds (note: negative values)
        }

        # Create feature matrix with proper ordering
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        # Scale each column based on its bounds
        for col, (min_val, max_val) in bounds.items():
            if col in X_train_scaled.columns:
                X_train_scaled[col] = (X_train_scaled[col] - min_val) / (max_val - min_val)
                X_test_scaled[col] = (X_test_scaled[col] - min_val) / (max_val - min_val)
                print(f"  {col}: scaled to [{min_val:.6f}, {max_val:.6f}]")

        X_train = X_train_scaled
        X_test = X_test_scaled

    print("\n" + "=" * 80)
    print("GAUSSIAN PROCESS TRAINING")
    print("=" * 80)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    # print(f"Number of targets: {y_train.shape[1]}")

    # Define kernel
    if kernel_type == "rbf":
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        print(f"Kernel type: RBF")
    elif kernel_type == "matern":
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5)
        print(f"Kernel type: Matern (nu=2.5)")
    elif kernel_type == "linear":
        kernel = C(1.0, (1e-3, 1e3)) * DotProduct()
        print(f"Kernel type: Linear")
    elif kernel_type == "linear_rbf":
        kernel = C(1.0, (1e-3, 1e3)) * (DotProduct() + RBF(1.0, (1e-2, 1e2)))
        print(f"Kernel type: Linear + RBF")
    elif kernel_type == "linear_matern":
        kernel = C(1.0, (1e-3, 1e3)) * (DotProduct() + Matern(nu=2.5))
        print(f"Kernel type: Linear + Matern (nu=2.5)")
    else:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        print(f"Kernel type: RBF (default)")

    # Train Gaussian Process
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=30,
        alpha=1e-6,
        normalize_y=True,
        random_state=random_state,
    )

    print("\nFitting Gaussian Process...")
    gp.fit(X_train, y_train)
    print("Fitting complete!")

    # Make predictions
    y_pred_train, y_std_train = gp.predict(X_train, return_std=True)
    y_pred_test, y_std_test = gp.predict(X_test, return_std=True)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(f"\nTraining Set:")
    print(f"  MSE:  {train_mse:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  MAE:  {train_mae:.6f}")
    print(f"  R²:   {train_r2:.6f}")

    print(f"\nTest Set:")
    print(f"  MSE:  {test_mse:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE:  {test_mae:.6f}")
    print(f"  R²:   {test_r2:.6f}")

    # Save results to CSV
    results_df = pd.DataFrame(
        {
            "y_actual": y_test,
            "y_predicted": y_pred_test,
            "y_std": y_std_test,
            "residuals": y_test - y_pred_test,
            "v": X_test["v"].values,
            "CO2share": X_test["CO2share"].values,
            "E_appl": X_test["E_appl"].values,
        }
    )

    results_file = os.path.join(output_dir, "gp_predictions.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Save metrics to file
    metrics_file = os.path.join(output_dir, "gp_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("GAUSSIAN PROCESS REGRESSION METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target variable: {output_col}\n")
        f.write(f"Kernel type: {kernel_type}\n")
        f.write(f"Training set size: {len(X_train)}\n")
        f.write(f"Test set size: {len(X_test)}\n\n")
        f.write("TRAINING SET:\n")
        f.write(f"  MSE:  {train_mse:.6f}\n")
        f.write(f"  RMSE: {train_rmse:.6f}\n")
        f.write(f"  MAE:  {train_mae:.6f}\n")
        f.write(f"  R²:   {train_r2:.6f}\n\n")
        f.write("TEST SET:\n")
        f.write(f"  MSE:  {test_mse:.6f}\n")
        f.write(f"  RMSE: {test_rmse:.6f}\n")
        f.write(f"  MAE:  {test_mae:.6f}\n")
        f.write(f"  R²:   {test_r2:.6f}\n")
    print(f"Metrics saved to: {metrics_file}")

    return {
        "model": gp,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_std_test": y_std_test,
        "metrics": {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
        },
    }


def plot_confidence_bands(
    model: GaussianProcessRegressor,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    output_dir: str = None,
    bounds: dict = None,
) -> None:
    """
    Create plots showing GP predictions with confidence bands for each input variable.

    Parameters:
    -----------
    model : GaussianProcessRegressor
        Trained Gaussian Process model
    X_train : pd.DataFrame
        Training input features
    y_train : np.ndarray
        Training target values
    output_dir : str
        Directory to save plots
    bounds : dict
        Dictionary with bounds for each variable (min, max)
    """

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "GP_runs")
    os.makedirs(output_dir, exist_ok=True)

    if bounds is None:
        bounds = {"v": (0, 1), "CO2share": (0, 1), "E_appl": (0, 1)}  # scaled bounds

    import matplotlib.pyplot as plt

    input_features = ["v", "CO2share", "E_appl"]

    for feature_idx, feature_name in enumerate(input_features):
        print(f"\nCreating confidence band plot for {feature_name}...")

        # Create a grid of points for this feature
        n_points = 100
        feature_range = np.linspace(bounds[feature_name][0], bounds[feature_name][1], n_points)

        # Use mean values for other features
        X_grid = np.zeros((n_points, X_train.shape[1]))

        for i, val in enumerate(feature_range):
            # Set the current feature value
            X_grid[i, feature_idx] = val

            # Set other features to their mean values (scaled)
            for j in range(X_train.shape[1]):
                if j != feature_idx:
                    X_grid[i, j] = X_train.iloc[:, j].mean()

        # Make predictions with uncertainty
        y_pred, y_std = model.predict(X_grid, return_std=True)

        # Create confidence bands (±1 std)
        y_lower = y_pred - y_std
        y_upper = y_pred + y_std

        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot training data
        ax.scatter(
            X_train.iloc[:, feature_idx],
            y_train,
            color="red",
            s=30,
            alpha=0.5,
            label="Training data",
            edgecolors="darkred",
        )

        # Plot mean prediction
        ax.plot(feature_range, y_pred, "b-", linewidth=2, label="GP mean prediction")

        # Plot confidence band (±1 std)
        ax.fill_between(feature_range, y_lower, y_upper, alpha=0.2, color="blue", label="±1 std (confidence band)")

        # Plot confidence band (±2 std)
        y_lower_2 = y_pred - 2 * y_std
        y_upper_2 = y_pred + 2 * y_std
        ax.fill_between(feature_range, y_lower_2, y_upper_2, alpha=0.1, color="blue", label="±2 std")

        ax.set_xlabel(feature_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("CO2_out", fontsize=12, fontweight="bold")
        ax.set_title(f"GP Prediction with Confidence Bands: CO2_out vs {feature_name}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"confidence_band_{feature_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {plot_path}")
        plt.show()
        plt.close()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_std_test": y_std_test,
        "metrics": {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
        },
    }


if __name__ == "__main__":
    project_name = "run_linear_maternkernelscaled"

    # Load data
    X_train, Y_train = prepare_training_data(
        matlab_folder=r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\4k"
    )
    X_test, Y_test = prepare_training_data(
        matlab_folder=r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\RandomData"
    )

    # Train and test Gaussian Process
    results = train_test_gaussian_process(
        X_train=X_train,
        X_test=X_test,
        y_train=Y_train["CO2_out"],
        y_test=Y_test["CO2_out"],
        kernel_type="linear_matern",
        output_dir="GP_runs/" + project_name,
    )

    # Create confidence band plots
    output_dir = os.path.join(os.path.dirname(__file__), "GP_runs/" + project_name)
    plot_confidence_bands(
        model=results["model"],
        X_train=results["X_train"],
        y_train=results["y_train"],
        output_dir=output_dir,
        bounds={"v": (0, 1), "CO2share": (0, 1), "E_appl": (0, 1)},
    )
