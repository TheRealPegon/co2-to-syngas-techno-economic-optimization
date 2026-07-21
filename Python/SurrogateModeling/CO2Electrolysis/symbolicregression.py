import pandas as pd
import os
from pysr import PySRRegressor, TensorBoardLoggerSpec
import numpy as np
from Python.SurrogateModeling.CO2Electrolysis.prepare_training_data import prepare_training_data
import numpy as np
from Python.SurrogateModeling.CO2Electrolysis.symbolicregression_testing import test_symbolic_regression
from pathlib import Path


def train_symbolic_regression(X, Y, runname="testrun"):
    # Create path to SR_runs folder to save output"
    sr_run_dir = os.path.join(os.path.dirname(__file__), "SR_runs", runname)

    # Create a logger that writes to "logs/run*":
    logger_spec = TensorBoardLoggerSpec(
        log_dir="logs/run",
        log_interval=10,  # Log every 10 iterations
    )

    model = PySRRegressor(
        elementwise_loss="L1DistLoss()",
        niterations=4000,
        maxsize=30,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[
            "square",
            "cube",
            "sqrt",
            "cbrt",
            "exp",
            "log",
         ],
        constraints={"^": (-1, 2)},
        nested_constraints={
            "square": {"square": 0, "cube": 0, "exp": 0, "sqrt": 0},
            "cube": {"square": 0, "cube": 0, "exp": 0, "cbrt": 0},
            "exp": {"log": 0, "square": 0, "cube": 0, "exp": 0},
            "log": {"exp": 0, "log": 0, "sqrt": 0, "cbrt": 0},
         },
        logger_spec=logger_spec,
        output_directory=sr_run_dir,
        procs=4,  # Reduced from 8 to avoid OOM
        parallelism="multiprocessing",
        model_selection="best",
        warm_start=False,
        turbo=True,
        parsimony=0.01,
    )

    model.fit(X, Y)
    return model


if __name__ == "__main__":
    # AFTER TRAINING: Bash> tensorboard --logdir=logs/run
    # THEN OPEN BROWSER at: http://localhost:6006/
    rstate = np.random.RandomState(42)
    target = "CO2_out"

    X, Y, df = prepare_training_data(
        matlab_folder=r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\LHS_data"
    )

    X_test, Y_test, df_test = prepare_training_data(
        matlab_folder=r"c:\Users\Philippe\Documents\UniCours\Master\Masterarbescht\MasterCode\master-thesis-orlewski\Matlab\RandomData"
    )

    model = train_symbolic_regression(X, Y[target], runname=target)
    pickle_path = Path(model.output_directory) / str(model.run_id_) / "checkpoint.pkl"

    test_symbolic_regression(
        X_test,
        Y_test,
        "best",
        pickle_path,
        y=target,
    )

    test_symbolic_regression(
        X_test,
        Y_test,
        "score",
        pickle_path,
        y=target,
    )

    test_symbolic_regression(
        X_test,
        Y_test,
        "accuracy",
        pickle_path,
        y=target,
    )
