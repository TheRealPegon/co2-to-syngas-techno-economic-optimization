import numpy as np
import pandas as pd
import os
import pickle
from SurrogateModeling.CO2Electrolysis.prepare_training_data import prepare_training_data
import xlsxwriter


def test_symbolic_regression(X_test, Y_test, model_selection, pickle_path, y):
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)

    model.model_selection = model_selection
    chosen_equation = model.get_best()["equation"]

    # Print model equation
    print("\n" + "=" * 80)
    print("MODEL EQUATION")
    print("=" * 80)

    print(model)

    # Make predictions
    Y_pred = model.predict(X_test.values)
    # Calculate L2 error (RMSE)
    Y_test_values = Y_test[y].values
    l2_error = np.sqrt(np.mean((Y_pred - Y_test_values) ** 2))

    print("\n" + "=" * 80)
    print("SYMBOLIC REGRESSION TEST RESULTS")
    print("=" * 80)
    print(f"L2 Error (RMSE): {l2_error:.6f}")
    print(f"Number of test samples: {len(Y_pred)}")
    print(f"Mean Prediction: {np.mean(Y_pred):.6f}")
    print(f"Mean Actual: {np.mean(Y_test_values):.6f}")
    print("=" * 80)

    # Calculate difference
    difference = Y_pred - Y_test_values
    percent_difference = abs((difference / Y_test_values)) * 100
    # Create DataFrame with X_test, predictions, actual values, and difference
    results_df = pd.DataFrame(X_test)
    results_df["Vcell"] = Y_test["Vcell"]
    results_df["CO_out"] = Y_test["CO_out"]
    results_df["H2_out"] = Y_test["H2_out"]
    results_df["H2_out"] = Y_test["CO2_out"]
    results_df["Predicted"] = Y_pred
    results_df["Actual"] = Y_test_values
    results_df["Difference"] = difference
    results_df["Absolute_Error"] = np.abs(difference)
    results_df["Percent_Difference"] = percent_difference
    # Write to CSV file
    file_name = model_selection + "_" + "symbolic_regression_predictions.xlsx"
    output_path = os.path.join(pickle_path, "..", file_name)
    # results_df.to_csv(output_path, index=False)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # Sheet 1: predictions
        results_df.to_excel(writer, sheet_name="Predictions", index=False)

        # Sheet 2: chosen equation
        equation_df = pd.DataFrame({"Chosen Equation": [chosen_equation]})
        equation_df.to_excel(writer, sheet_name="Model_Equation", index=False)
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    rstate = np.random.RandomState(42)
    X_test, Y_test, df_test = prepare_training_data(
        matlab_folder=r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\Random_betterbounds"
    )

    test_symbolic_regression(
        X_test,
        Y_test,
        "score",
        r"C:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Python\SurrogateModeling\SR_runs\H2_out\20260202_144845_Fmvw2a\checkpoint.pkl",
        y="H2_out",
    )
