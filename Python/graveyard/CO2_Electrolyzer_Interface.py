import numpy as np
import matlab.engine
import os
import time
from pathlib import Path


from typing import (
    Any,
    Dict,
)


class CO2_Electrolyzer_Interface:
    """Interface to run CO2 electrolyzer simulations using MATLAB engine."""

    # --- Instance Attribute Type Hints ---
    eng: Any  # Keep Any for the external MATLAB engine object
    matlab_results: Dict[str, Any]

    execution_time: float

    # Unpacked results
    X: Dict[str, float]
    FE: float
    c_outlet_CO2: np.ndarray
    c_outlet_CO: np.ndarray
    c_outlet_H2: np.ndarray
    delP: float
    CD: float
    Vcell: float

    def __init__(
        self,
        relative_path="../Matlab",
    ):
        current_file = Path(__file__)
        matlab_path = (current_file / ".." / relative_path).resolve()
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(
            str(matlab_path),
            nargout=0,
        )

    def run_simulation(
        self,
        E_applied,
        c_inlet,
    ):
        """Takes python variables, converts them to matlab types and then calls the matlab wrapper file M3_CO_wrapper.m

        Args:
            E_applied (float): Applied cell potential in volts (NO THE ACTUAL VOLTAGE! THAT'S AN OUTPUT OF THE MODEL) [V]
            c_inlet (list[float]): list of inlet concentrations for CO2, CO, and H2 in mol/m3 in that order
        """
        start_time = time.time()

        # Convert Python types to matlab types
        c_inlet_matlab = matlab.double([c_inlet])
        E_applied_matlab = matlab.double(E_applied)

        # Call the MATLAB function for CO2 electrolyzer simulation
        self.matlab_results = self.eng.M3_CO_wrapper(
            E_applied_matlab,
            c_inlet_matlab,
        )

        self.check_results()

        # Close the MATLAB engine
        self.eng.quit()

        self.execution_time = time.time() - start_time

        return self

    def check_results(
        self,
    ):
        """Unpacks matlab results and losely checks for invalidity of simulation then throws Value Error if it does

        Raises:
            ValueError: CO2 concentration in change in the column too large
            ValueError: CO concentration in change in the column too large
            ValueError: H2 concentration in change in the column too large
            ValueError: Any concentration < 0
            ValueError: Total, Heterogeneous or Homogeneous conversion below zero
        """
        self.X = self.matlab_results["X"]
        self.FE = self.matlab_results["FE"]
        self.c_outlet_CO2 = np.array(self.matlab_results["y"]["CO2"][0]).reshape(-1)
        self.c_outlet_CO = np.array(self.matlab_results["y"]["C2H4"][0]).reshape(-1)
        self.c_outlet_H2 = np.array(self.matlab_results["y"]["H2"][0]).reshape(-1)
        self.delP = self.matlab_results["delP"]
        self.CD = self.matlab_results["CD"]
        self.Vcell = self.matlab_results["Vcell"]

        tolerance = 100  # 10000% tolerance
        for i in range(
            1,
            len(self.c_outlet_CO2) - 1,
        ):
            percent_change_CO2 = abs((self.c_outlet_CO2[i + 1] - self.c_outlet_CO2[i]) / self.c_outlet_CO2[i])
            percent_change_CO = abs((self.c_outlet_CO[i + 1] - self.c_outlet_CO[i]) / self.c_outlet_CO[i])
            percent_change_H2 = abs((self.c_outlet_H2[i + 1] - self.c_outlet_H2[i]) / self.c_outlet_H2[i])
            if percent_change_CO2 > tolerance:
                raise ValueError(f"Large change in c_outlet_CO2 at index {i}. Percent change: {percent_change_CO2:.2%}")
            if percent_change_CO > tolerance:
                raise ValueError(f"Large change in c_outlet_CO at index {i}. Percent change: {percent_change_CO:.2%}")
            if percent_change_H2 > tolerance:
                raise ValueError(f"Large change in c_outlet_H2 at index {i}. Percent change: {percent_change_H2:.2%}")
        if np.any(self.c_outlet_CO2 < 0):
            raise ValueError("CO2 concentration is below zero.")
        if np.any(self.c_outlet_CO < 0):
            raise ValueError("CO concentration is below zero.")
        if np.any(self.c_outlet_H2 < 0):
            raise ValueError("H2 concentration is below zero.")

        for key in self.X:
            if self.X[key] < 0:
                raise ValueError(f"Conversion {key} is below zero: {self.X[key]:.2f}")
