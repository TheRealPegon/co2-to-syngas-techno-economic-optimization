from Python.graveyard.flowsheet_builder import FlowsheetModel

if __name__ == "__main__":
    flowsheet = FlowsheetModel()
    solver = flowsheet.minimize_OPEX()
    flowsheet.display_results()
    print(solver.solver.termination_condition)
    # Extract iterations from the solver's metadata
    iterations = solver.solver.statistics.get("number_of_iterations", "N/A")
    stats = solver.solver.statistics
    solver.write()
    print(f"Total Iterations: {iterations}")
