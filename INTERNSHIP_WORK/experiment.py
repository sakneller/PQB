import csv
import datetime
import json
import os
import pickle

import beautifultable as bt
import pandas as pd

import dimod
from dimod import ConstrainedQuadraticModel, Integer
from dwave.system import LeapHybridCQMSampler
from neal import SimulatedAnnealingSampler

import numpy as np


#8/3/23 ------

import sys

dataInput = input("What is the name of your data file?")
dataFile = "/workspace/PQB/INTERNSHIP_WORK/"+dataInput
if os.path.isfile(dataFile) == False:
    print("Sorry, the data file you specified does not exist!")
    sys.exit(0)

outputFileName = input("What do you want the file name of your output file to be?")
outputFile = "/workspace/PQB/INTERNSHIP_WORK/output/"+outputFileName
print(outputFile)
#if os.path.isfile("outputFile") == True:
#    print("Sorry, the output file you specified already exists!")
#    sys.exit(0)

test = input("Do you want to keep running the program?")
if test.title() == "No":
    sys.exit(0)
    

#-------


def create_binary_vars(df):
    """
    Creates a column in `df` containing a dimod.Binary object for each binary
    decision variable (one for each row in `df`) and another column for their
    label name

    df : pandas.DataFrame
    """

    df["x_values"] = list(f"x{idx}" for idx in df.index)
    df["binary_obj"] = df.apply(lambda row : dimod.Binary(row["index"]), axis=1)


def read_and_format(csv_path):
    """
    Reads-in and format CSV for processing
    Returns: pandas.DataFrame

    csv_path: str
        path of CSV containing data
    """

    # store data in dataframe
    df = pd.read_csv(csv_path, skipinitialspace=True)

    # Format CSV
    df = df.rename(lambda name : name.strip(), axis="columns")
    df["index"] = df.index
    # Remove dollar-sign and convert price to float

    #Format of how to change values in csv table if necessary
    #df["price"] = df.apply(lambda row : float(row["price"].replace("$", "").strip()), axis=1)

    # insert columns for binary decision variables and dimod.Binary objects
    create_binary_vars(df)

    return df



def format_objective(df, col_name):
    """
    Formats the objective function for pretty-printing
    Returns: str

    df : pandas.DataFrame
    col_name : str
        name of column in `df` that containing values to use in building the
        objective function (ie: the linear coefficients)
    """

    result = df[col_name].map(str) + df["x_values"]
    return " + ".join(result)


def format_one_hot(df, col_name):
    """
    Formats the one-hot constraints for pretty-printing
    Returns: str

    df : pandas.DataFrame
    col_name : str
        name of column in `df` that contains item categories
    """

    result = ""
    for category in df[col_name].unique():
        result += " + ".join(df[df[col_name] == category]["x_values"]) + " = 1\n"
    return result


def format_constraints(df, info):
    """
    Formats the constraints for pretty-printing
    Returns: str

    df : pandas.DataFrame
    constraints: list of dict
        (in)equality constraints, items should have the following keys:
        - col_names: names of columns containing coefficients in dataframe
        - operators: "=", "<=", ">=", ">", "<" or "="
        - comparison_values: right-hand side values of constraints
    """

    result_inequality = ""
    """for item in info:
        terms = " + ".join(df[item["col_name"]].map(str) + df["x_values"])
        op = item["operator"]
        value = str(item["comparison_value"])
        result_inequality += " ".join([terms, op, value]) + "\n" """
    return result_inequality


def show_cqm(objective, constraints):
    """
    Formats the objective and constraints for pretty-printing
    Returns: str

    objective : str
        the objective function
    constraints : list of str
        the constraints
    """

    return "Minimize " + objective + "\n\nConstraints\n" + "\n".join(constraints)


def create_cqm_model():
    """
    Initialize the CQM object
    Returns: dimod.ConstrainedQuadraticModel
    """

    return dimod.CQM()


def define_objective(df, cqm, col_name):
    """
    Sets the objective function for the CQM model

    df : pandas.DataFrame
    cqm : dimod.ConstrainedQuadraticModel
    col_name : str
        name of column in `df` that containing values to use in building the
        objective function (ie: the linear coefficients)
    """

    objective_terms = df[col_name] * df["binary_obj"]
    cqm.set_objective(sum(objective_terms))



def define_one_hot(df, cqm, col_name):
    """
    Applies one-hot constraints to the CQM

    df : pandas.DataFrame
    cqm : dimod.ConstrainedQuadraticModel
    col_name : str
        name of column in `df` that contains item categories
    """

    for category in df[col_name].unique():
        constraint_terms = list(df[df["leg"] == category]["binary_obj"])
        cqm.add_discrete(sum(constraint_terms))


def define_one_hot_new(df, cqm, col_name):
    """
    Applies one-hot constraints to the CQM

    df : pandas.DataFrame
    cqm : dimod.ConstrainedQuadraticModel
    col_name : str
        name of column in `df` that contains item categories
    """

    #for category in df[col_name].unique():
    #    constraint_terms = list(df[df["leg"] == category]["binary_obj"])
    #    cqm.add_discrete(sum(constraint_terms))

    #x, y = dimod.Binaries(df.at[1, "binary_obj"], df.at[2, "binary_obj"])
    #cqm = dimod.ConstrainedQuadraticModel()
    #cqm.add_constraint(x + y == 1, label='ABC_path')
    #cqm.constraints['ABC_path '].to_polystring()

    #cqm = dimod.ConstrainedQuadraticModel()
    #x,y = dimod.Binaries([df.at[1, "binary_obj"], df.at[1, "binary_obj"]])
    #cqm.add_discrete(sum([x,y]), label="ABC_path")

    modes = [key for key in method.keys() if method[key]["use"]]
    num_modes = len(modes)

    path = [key for key in path.keys() if path[key]["use"]]
    num_paths = len(path)
    t= [dimod.Binary(f"{mode}_{i}") for i in range(num_paths) for mode in modes]

    #cqm = dimod.ConstrainedQuadraticModel()

    #cqm.set_objective(-_calculate_total(t, "Exercise", legs, locomotion_vals))
    paths_total = sum(l["length"] for p in paths)
    max_env_cost = round(paths_total * np.mean([min(env_cost), max(env_cost)]))

    for leg in range(num_paths):
        cqm.add_constraint(dimod.quicksum(t[num_modes*leg:num_modes*leg+num_modes]) == 1,
            label=f"One-hot path{leg}")
    cqm.add_constraint((t, "Environmental Cost", paths, locomotion_vals) <= max_env_cost,
        label="Total cost",
        weight=weight_vals["weight_cost"]["weight"],
        penalty=weight_vals["weight_cost"]["penalty"])
    cqm.add_constraint((t, "Time", num_path, locomotion_vals) <= 60,
        label="Total time",
        weight=weight_vals["weight_time"]["weight"],
        penalty=weight_vals["weight_time"]["penalty"])
    
    


def define_constraints(df, cqm, constraints):
    """
    Applies inequality constraints to the CQM

    df : pandas.DataFrame
    cqm : dimod.ConstrainedQuadraticModel
    constraints: list of dict
        (in)equality constraints, items should have the following keys:
        - col_names: names of columns containing coefficients in dataframe
        - operators: "=", "<=", ">=", ">", "<" or "="
        - comparison_values: right-hand side values of constraints
    """

    for item in constraints:
        constraint_terms = df[item["col_name"]] * df["binary_obj"]

        if item["operator"] == "<=":
            cqm.add_constraint(sum(constraint_terms) <= item["comparison_value"])
        elif item["operator"] == ">=":
            cqm.add_constraint(sum(constraint_terms) >= item["comparison_value"])
        elif item["operator"] == "<":
            cqm.add_constraint(sum(constraint_terms) < item["comparison_value"])
        elif item["operator"] == ">":
            cqm.add_constraint(sum(constraint_terms) > item["comparison_value"])
        elif item["operator"] == "=" or item["operator"] == "==":
            cqm.add_constraint(sum(constraint_terms) == item["comparison_value"])
        else:
            raise Exception(f"Your choice of {item['comparison_value']} is invalid. Choose from '>', '<', '>=', '<=' or '='")


def print_cqm_stats(cqm):
    """
    Print some information about the CQM model defining the 3D bin packing problem.

    cqm : dimod.ConstrainedQuadraticModel
        should be set up using `create_cqm_model`, `define_objective`,
        `define_one_hot` and/or `define_constraints`
    """

    if not isinstance(cqm, dimod.ConstrainedQuadraticModel):
        raise ValueError("input instance should be a dimod CQM model")

    # collect info about our variables
    vartypes = [cqm.vartype(v).name for v in cqm.variables]
    num_binaries = sum(1 for v in vartypes if v == "BINARY")
    num_integers = sum(1 for v in vartypes if v == "INTEGER")
    num_continuous = sum(1 for v in vartypes if v == "REAL")
    num_discretes = len(cqm.discrete)
    # collect info about our constraints
    num_linear_constraints = sum(
        constraint.lhs.is_linear()
        for constraint in cqm.constraints.values()
    )
    num_quadratic_constraints = sum(
        not constraint.lhs.is_linear()
        for constraint in cqm.constraints.values()
    )
    num_le_inequality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Le
        for constraint in cqm.constraints.values()
    )
    num_ge_inequality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Ge
        for constraint in cqm.constraints.values()
    )
    num_equality_constraints = sum(
        constraint.sense is dimod.sym.Sense.Eq
        for constraint in cqm.constraints.values()
    )

    assert num_binaries == len(cqm.variables)

    assert (
        num_quadratic_constraints + num_linear_constraints
        == len(cqm.constraints)
    )

    subtables = []
    # variables subtable
    subtables.append(bt.BeautifulTable())
    subtables[-1].columns.header = ["Binary", "Integer", "Continuous"]
    subtables[-1].rows.append([num_binaries, num_integers, num_continuous])
    # constraints subtable
    subtables.append(bt.BeautifulTable())
    subtables[-1].columns.header = ["Quadratic", "Linear", "One-Hot"]
    subtables[-1].rows.append([num_quadratic_constraints, num_linear_constraints, num_discretes])
    # sensitivity subtable
    subtables.append(bt.BeautifulTable())
    subtables[-1].columns.header = ["EQ", "LT", "GT"]
    subtables[-1].rows.append(
        [num_equality_constraints - num_discretes, num_le_inequality_constraints, num_ge_inequality_constraints]
    )
    # subtable styling
    for tbl in subtables:
        tbl.set_style(bt.STYLE_COMPACT)

    # merge subtables
    full_table = bt.BeautifulTable(maxwidth=120)
    full_table.columns.header = ["Variables", "Constraints", "Sensitivity"]
    full_table.rows.append(subtables)
    full_table.set_style(bt.STYLE_COMPACT)

    title = "MODEL INFORMATION"
    # for title centering
    padding = len(str(full_table).split("\n", maxsplit=1)[0]) - len(title) - 2
    print("=" * (padding // 2), title, "=" * (padding // 2 + padding % 2))
    print(full_table)


def cqm_sample(cqm, label):
    """
    Samples the CQM model and filters out non-feasible solutions
    Returns: dimod.SampleSet

    cqm : dimod.ConstrainedQuadraticModel
        should be set up using `create_cqm_model`, `define_objective`,
        `define_one_hot` and/or `define_constraints`
    label : str
        name to associate with the CQM run when submitting to DWave
    """

    print("\nSolving CQM\n")
    if int(os.environ.get("DEBUG", 0)) == 1:
        # use simulated annealing when in debug mode
        # this can be very slow and give sub-optimal results compared to using the LeapHybridCQMSampler,
        # so it is best to only use this on small data sets for testing purposes.
        sampler = SimulatedAnnealingSampler()
        bqm, invert = dimod.cqm_to_bqm(cqm)
        sample_set = sampler.sample(bqm, num_reads=int(os.environ.get("DEBUG_NUMREADS", 100)))
        sample_set = dimod.SampleSet.from_samples_cqm([invert(x) for x in sample_set.samples()], cqm)
    else:
        # default to using the LeapHybridCQMSampler
        sampler = LeapHybridCQMSampler()
        time_limit = sampler.min_time_limit(cqm)
        sample_set = sampler.sample_cqm(cqm, label=label, time_limit=time_limit)

    # filter out non-feasible samples and warn if no solution was found
    feasible_sampleset = sample_set.filter(lambda row: row.is_feasible)
    if len(feasible_sampleset) < 1:
        print("No solution meeting constraints found")
        return None

    return feasible_sampleset


def format_cqm_results(df, results):
    """
    Formats the CQM results into 2d-array (for easily parsing to CSV etc).
    Returns: list of lists

    df : pandas.DataFrame
    results : dimod.SampleSet
        filtered CQM results, as returned by `cqm_sample`
    """

    formatted_results = []
    imp_column_names = df.columns[2:-5]
    header = list(df["leg"].unique()) + list(imp_column_names) + ["energy", "num_occ"]
    formatted_results.append(header)

    # if no feasible solutions were found,
    if results is None:
        return formatted_results

    # otherwise, build the data table
    for record in results.record:
        sample = record["sample"]
        x_indicies = [index for index, value in enumerate(sample) if value == 1]
        x_values = [results.variables[index] for index in x_indicies]
        decision_data_df = df.loc[df["index"].isin(x_values)]
        samples = []
        num_of_legs = len(df["leg"].unique())

        for leg in df["leg"].unique():
            item_index_sample = decision_data_df.loc[decision_data_df["leg"] == leg, "method"]
            item_index_sample_two = decision_data_df.loc[decision_data_df["leg"] == leg, "env_cost"]
            if (len(item_index_sample) != 1) or (len(item_index_sample_two) != 1):
                print("One-Hot constraint not met")
                continue
            samples.append(item_index_sample.to_list()[0])

        if len(samples) != num_of_legs:
            print("Number of types of items should equal length of samples list")
            continue

        samples.extend([str(sum(decision_data_df[column_name])) for column_name in imp_column_names])
        samples.append(str(record["energy"]))
        samples.append(str(record["num_occurrences"]))
        formatted_results.append(samples)

    return formatted_results


def save_cqm_results(cqm, solutions, formatted_results, label):
    """
    Save the results of sampling run.
    Returns: (str) path to the saved sample set

    cqm : dimod.ConstrainedQuadraticModel
        CQM model object
    solutions : dimod.SampleSet
        filtered CQM results, as returned by `cqm_sample`
    formatted_results : str
        formatted output of CQM, as returned by `format_cqm_results`
    """

    folder = os.path.join(outputFile, label)
    if not os.path.exists(folder):
        os.makedirs(folder)

    serializable_results = solutions.to_serializable()
    with open(os.path.join(folder, "sample_set.pkl"), "wb") as f:
        pickle.dump(serializable_results, f)

    model_path = os.path.join(folder, "CQM_model")
    with open(model_path, "wb") as f:
        cqm_file = cqm.to_file()
        cqm_contents = cqm_file.read()
        f.write(cqm_contents)

    with open(os.path.join(folder, "human_readable_results.csv"), "w") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerows(formatted_results)

    return os.path.join(folder, "sample_set.pkl")


def load_results(results_path):
    """
    Reads in stored results of a sampling run.
    Returns: (dimod.SampleSet) the set of feasible solutions

    results_path : str
        path to stored feasible solutions, as JSON or python pickle
    """

    if results_path.lower().endswith(".json"):
        with open(os.path.join(results_path), "r") as fh:
            sample_set = json.load(fh)
    elif results_path.lower().endswith(".pkl"):
        with open(os.path.join(results_path), "rb") as fh:
            sample_set = pickle.load(fh)
    else:
        ext = os.path.splitext(results_path)[1]
        raise NotImplementedError(f"Unsure how to process file extension {ext}")

    loaded_sample_set = dimod.SampleSet.from_serializable(sample_set)
    return loaded_sample_set


def print_hamiltonian(df, objective_column, one_hot_column, constraints):
    """
    Formats and prints the Hamiltonian corresponding to the CQM defined by the
    arguments.

    df : pandas.DataFrame
        the input data, as loaded using `read_and_format`
    objective_column: str
        the column name to use as the objective to minimize
    one_hot_column: str
        the column name to use as the one-hot constraint
    constraints: list of dict
        (in)equality constraints, items should have the following keys:
        - col_names: names of columns containing coefficients in dataframe
        - operators: "=", "<=", ">=", ">", "<" or "="
        - comparison_values: right-hand side values of constraints
    """

    # Get objective string
    objective = format_objective(df, objective_column)

    # Get one-hot constraint string
    #one_hot = format_one_hot(df, one_hot_column)

    # Get constraint string
    ineq = format_constraints(df, constraints)

    # Format and print QUBO
    #print(show_cqm(objective, [one_hot, ineq]))
    print(show_cqm(objective, [ineq]))


def resolve_cqm(df, objective_column, one_hot_column, constraints, label):
    """
    Builds and solves the CQM model as defined by the arguments. Also formats
    and saves the results from DWave.
    Returns: (str) path to the CQM results pickle (see also: `save_cqm_results`)

    df : pandas.DataFrame
        the input data, as loaded using `read_and_format`
    objective_column: str
        the column name to use as the objective to minimize
    one_hot_column: str
        the column name to use as the one-hot constraint
    constraints: list of dict
        (in)equality constraints, items should have the following keys:
        - col_names: names of columns containing coefficients in dataframe
        - operators: "=", "<=", ">=", ">", "<" or "="
        - comparison_values: right-hand side values of constraints
    label : str
        name to associate with the CQM run when submitting to DWave
    """

    # Initialize CQM model
    cqm_model = create_cqm_model()

    # Set objective
    define_objective(df, cqm_model, objective_column)

    # Set one-hot constraints
    #define_one_hot(df, cqm_model, one_hot_column)

    #new one-hot function to set one-hot constraints
    #define_one_hot(df, cqm_model, one_hot_column)

    # Set inequality constraints
    define_constraints(df, cqm_model, constraints_1)
    define_constraints(df, cqm_model, CONSTRAINTS_2)
    define_constraints(df, cqm_model, CONSTRAINTS_3)
    # Print additional CQM information
    print_cqm_stats(cqm_model)

    # Initiate sampler
    feasible_solutions = cqm_sample(cqm_model, label)

    # Format CQM results
    formatted_results = format_cqm_results(df, feasible_solutions)

    # Save results and model
    return save_cqm_results(cqm_model, feasible_solutions, formatted_results, label)


def solve_cqm(input_csv, objective_column, one_hot_column, constraints, label):
    """
    Builds and solves the CQM model as defined by the arguments. Also prints
    the QUBO Hamiltonian and formatted CQM results.

    input_csv : str
        the path to the input data CSV
    objective_column: str
        the column name to use as the objective to minimize
    one_hot_column: str
        the column name to use as the one-hot constraint
    constraints: list of dict
        (in)equality constraints, items should have the following keys:
        - col_names: names of columns containing coefficients in dataframe
        - operators: "=", "<=", ">=", ">", "<" or "="
        - comparison_values: right-hand side values of constraints
    label : str
        name to associate with the CQM run when submitting to DWave
    """

    # Read in Data
    df = read_and_format(input_csv)
    print(df)
    # Print problem to screen
    print_hamiltonian(df, objective_column, one_hot_column, constraints)

    #sys.exit(0)

    # Create sampler and retrieve location of results
    result_loc = resolve_cqm(df, objective_column, one_hot_column, constraints, label)

    # Load saved results
    loaded_results = load_results(result_loc)

    # Print formatted results to screen
    data = format_cqm_results(df, loaded_results)
    table = bt.BeautifulTable(maxwidth=120)
    # add data to table
    table.columns.header = data[0]
    for row in data[1:]:
        table.rows.append(row)
    # style stuff
    table.columns.header.separator = "="
    # override padding so it's not so wide
    table.columns.padding_left = 0
    table.columns.padding_right = 0
    # col width for numeric columns

    #tmp = [8, 8, 8, 8, 8, 8]

    # use the rest of the space for meal items columns

    #table.columns.width = [(120 - sum(tmp)) // 5] * 5 + tmp
    table.columns.width = 12
    print(table)
    print()
    print(f"Results written to {result_loc.replace('./output', 'data')}")

#8/12/23 - 8/14/23
#def create_new_table(df):
#    df[leg]="AE"
#    length=0
#    time=0
#    cost=0
#    env_cost=0
#    segment_start="A"
#    segment_end="E"
#    while segment_end != "E":

        #from start point (A), check where we can move to in an if or while loop, and check if where we finish =E. create counter to check #of steps
        #if it finishes at E, then append row with updated characteristics to new table
        #

#        segment_end=end[randint(1,4)]
        #Add the length, time, and cost properties of segment chosen here
#        length+=
#        time+=
#        cost+=
#        env_cost+=
#        segment_end=
#        if segment_end=="E" then
#            .append()
            
    
#-------

if __name__ == "__main__":

    #INPUT_CSV = "/app/data/chicken_waffles_data.csv"
    INPUT_CSV = dataFile
    OBJECTIVE_COLUMN = "env_cost"
    ONE_HOT_COLUMN = "leg"
    constraints_1 = [{"col_name" : "time", "operator" : "<=", "comparison_value" : 60}]
    CONSTRAINTS_2 = [{"col name": "start_1", "operator" : "=", "comparison value": 1}]
    CONSTRAINTS_3 = [{"col name": "end_1", "operator" : "=", "comparison value": 5}]
    constraints=[(constraints_1,CONSTRAINTS_2,CONSTRAINTS_3)]
    # name to associate with DWave sampling run
    NAME = outputFileName
    # human-readable timestamp (in case of multiple runs)
    TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    solve_cqm(INPUT_CSV, OBJECTIVE_COLUMN, ONE_HOT_COLUMN, constraints, f"{NAME}-{TIMESTAMP}")
    #solve_cqm(INPUT_CSV, OBJECTIVE_COLUMN, ONE_HOT_COLUMN, CONSTRAINTS)