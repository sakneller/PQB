# Copyright 2014 Google Inc. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

"""Performs requests to the Google Maps Distance Matrix API."""


"""From the Intern:
Here we are importing all the necessary libraries for the CQM and Google Maps API
Distance Matrix to run."""


import googlemaps
from googlemaps import convert
import json
from json import *

import datetime as DT

import csv
import datetime
import os
import pickle

import beautifultable as bt
import pandas as pd

import dimod
from dimod import ConstrainedQuadraticModel, Integer
from dwave.system import LeapHybridCQMSampler
from neal import SimulatedAnnealingSampler

import numpy as np

import sys

import re

def distance_matrix(client, origins, destinations,
                    mode=None, language=None, avoid=None, units=None,
                    departure_time=None, arrival_time=None, transit_mode=None,
                    transit_routing_preference=None, traffic_model=None, region=None):
    """ Gets travel distance and time for a matrix of origins and destinations.

    :param origins: One or more addresses, Place IDs, and/or latitude/longitude
        values, from which to calculate distance and time. Each Place ID string
        must be prepended with 'place_id:'. If you pass an address as a string,
        the service will geocode the string and convert it to a
        latitude/longitude coordinate to calculate directions.
    :type origins: a single location, or a list of locations, where a
        location is a string, dict, list, or tuple

    :param destinations: One or more addresses, Place IDs, and/or lat/lng values
        , to which to calculate distance and time. Each Place ID string must be
        prepended with 'place_id:'. If you pass an address as a string, the
        service will geocode the string and convert it to a latitude/longitude
        coordinate to calculate directions.
    :type destinations: a single location, or a list of locations, where a
        location is a string, dict, list, or tuple

    :param mode: Specifies the mode of transport to use when calculating
        directions. Valid values are "driving", "walking", "transit" or
        "bicycling".
    :type mode: string

    :param language: The language in which to return results.
    :type language: string

    :param avoid: Indicates that the calculated route(s) should avoid the
        indicated features. Valid values are "tolls", "highways" or "ferries".
    :type avoid: string

    :param units: Specifies the unit system to use when displaying results.
        Valid values are "metric" or "imperial".
    :type units: string

    :param departure_time: Specifies the desired time of departure.
    :type departure_time: int or datetime.datetime

    :param arrival_time: Specifies the desired time of arrival for transit
        directions. Note: you can't specify both departure_time and
        arrival_time.
    :type arrival_time: int or datetime.datetime

    :param transit_mode: Specifies one or more preferred modes of transit.
        This parameter may only be specified for requests where the mode is
        transit. Valid values are "bus", "subway", "train", "tram", "rail".
        "rail" is equivalent to ["train", "tram", "subway"].
    :type transit_mode: string or list of strings

    :param transit_routing_preference: Specifies preferences for transit
        requests. Valid values are "less_walking" or "fewer_transfers".
    :type transit_routing_preference: string

    :param traffic_model: Specifies the predictive travel time model to use.
        Valid values are "best_guess" or "optimistic" or "pessimistic".
        The traffic_model parameter may only be specified for requests where
        the travel mode is driving, and where the request includes a
        departure_time.

    :param region: Specifies the prefered region the geocoder should search
        first, but it will not restrict the results to only this region. Valid
        values are a ccTLD code.
    :type region: string

    :rtype: matrix of distances. Results are returned in rows, each row
        containing one origin paired with each destination.
    """

    params = {
        "origins": convert.location_list(origins),
        "destinations": convert.location_list(destinations)
    }

    if mode:
        # NOTE(broady): the mode parameter is not validated by the Maps API
        # server. Check here to prevent silent failures.
        if mode not in ["driving", "walking", "bicycling", "transit"]:
            raise ValueError("Invalid travel mode.")
        params["mode"] = mode

    if language:
        params["language"] = language

    if avoid:
        if avoid not in ["tolls", "highways", "ferries"]:
            raise ValueError("Invalid route restriction.")
        params["avoid"] = avoid

    if units:
        params["units"] = units

    if departure_time:
        params["departure_time"] = convert.time(departure_time)

    if arrival_time:
        params["arrival_time"] = convert.time(arrival_time)

    if departure_time and arrival_time:
        raise ValueError("Should not specify both departure_time and"
                         "arrival_time.")

    if transit_mode:
        params["transit_mode"] = convert.join_list("|", transit_mode)

    if transit_routing_preference:
        params["transit_routing_preference"] = transit_routing_preference

    if traffic_model:
        params["traffic_model"] = traffic_model

    if region:
        params["region"] = region

    return client._request("/maps/api/distancematrix/json", params)


key="AIzaSyApCDXCL7ELIu4HsLET8KFdKTUNrt2n8yc"
client=googlemaps.Client(key)


"""From the Intern: 
The following function reads the POI_TABLE.txt table where the values for origins, destinations
and modes are stored and appends them to the origins, destinations, and modes lists."""


origins = []
destinations = []
modes = []

with open('/workspace/PQB/FINAL/POI_TABLE.txt','r') as f:
    lines = f.readlines()

for line in lines:
    if "origins" in line:
        words = re.split('= |"|\n', line)
        origins.append(words[2])

    if "destinations" in line:
        words = re.split('= |"|\n', line)
        destinations.append(words[2]) 

    if "modes" in line:
        words = re.split('= |"|\n', line)
        modes.append(words[2]) 

#print(origins)
#print(destinations)
#print(modes)

#sys.exit()


"""
From the Intern:
The following for loop creates the pandas dataframe, which is then output to the csv file maps_data.csv.
maps_data.csv is overwritten each time the program is runs. In the table, the column start is the origin,
the column end is the destination, and modes is the mode chosen. Leg is the origin and destination connected
with a hyphen, and one solution from each leg must be chosen, as defined by the one hot constraint in the
main function at the end of the program. Time and Cost are rough estimates of the duration in minutes 
and cost in dollars of the trip for each leg, and env_cost is the environmental cost, in terms of pounds
of CO2 released. This is nested in a try & except loop because sometimes taking transit between two cities
is not possible, as a bus or train route may not exist between the two cities. In that case, no row is created
for
"""


df = pd.DataFrame()
df2 = pd.DataFrame()

for o in origins:
    for d in destinations:
        for m in modes:
            temp= distance_matrix(client,origins=o,destinations=d, mode=m)

            level0 = ["distance"]
            level1 = ["value"]
            level2 = ["duration"]

            for l in level0:
                for item in level1:
                    for item2 in level2:
                        
                        try:
                            length_of_trip = temp["rows"][0]["elements"][0][l][item]
                            duration_of_trip = temp["rows"][0]["elements"][0][item2][item]

                            duration_in_minutes = round(temp["rows"][0]["elements"][0][item2][item] / 60 , 3)
                            length_in_miles = round(3.28084 * temp["rows"][0]["elements"][0][l][item] / 5280, 3)

                            env_cost=0
                            if m == "walking" or m == "bicycling":
                                env_cost = 0
                            elif m == "driving":
                                env_cost = round(length_in_miles * 0.88,3)
                            else:
                                env_cost = round(0.59 * length_in_miles, 3)

                            leg = o + " to " + d


                            cost=0
                            if m == "walking" or m == "bicycling":
                                cost = 0
                            elif m == "driving":
                                cost = round(length_in_miles * 0.66,3)
                            else:
                                cost = round(0.0625 * length_in_miles, 3)

                            new_row_list = { "start" : o, "end" : d, "modes" : m, "length" : length_in_miles, "time" : duration_in_minutes, "cost":cost, "env_cost" : env_cost , "leg": leg}
                            df = pd.concat([df, pd.DataFrame([new_row_list])], ignore_index=True)

                        except KeyError:
                            print("ERROR: VALUE UNDEFINED")

#sys.exit()
df.to_csv("/workspace/PQB/FINAL/data/maps_data.csv")



"""
From the Intern: This marks the end of the code taken from the Google Maps Distance_Matrix code that is
used in order to determine the exact distance between two points. From here on, the code was taken
from the PQB Chicken & Waffle code that is linked in the README file.
"""



dataFile = "/workspace/PQB/FINAL/data/maps_data.csv"


"""
From the Intern: This is a bit of sort-of redundant code to check if the csv table POI_TABLE
can be found. If not, the system tells the program to stop running.
"""


if os.path.isfile(dataFile) == False:
    print("Sorry, the data file you specified does not exist!")
    sys.exit(0)

outputFileName = input("What do you want the file name of your output file to be?")
outputFile = "/workspace/PQB/FINAL/output/"+outputFileName
print(outputFile)

test = input("Do you want to keep running the program?")
if test.title() == "No":
    sys.exit(0)


"""
From the Intern:
From here on, the majority of the code & comments were created by Polaris QB unless noted otherwise.
There are comments inside most of the function explaining the functionality of them.
Comments made by the intern will begin with "From the Intern."
"""


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
    for item in info:
        terms = " + ".join(df[item["col_name"]].map(str) + df["x_values"])
        op = item["operator"]
        value = str(item["comparison_value"])
        result_inequality += " ".join([terms, op, value]) + "\n "
    return result_inequality

def show_cqm(objective, CONSTRAINTS):
    """
    Formats the objective and constraints for pretty-printing
    Returns: str

    objective : str
        the objective function
    constraints : list of str
        the constraints
    """

    return "Minimize " + objective + "\n\nConstraints\n" + "\n".join(CONSTRAINTS)

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


"""
From the Intern:
The following function ensures that only one solution is chosen for each 
leg, or origin-destination pair.
"""


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

def define_constraints(df, cqm, CONSTRAINTS):
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
    
    for item in CONSTRAINTS:
        print(item)
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
        [num_equality_constraints - num_discretes, num_le_inequality_constraints, CONSTRAINTS]
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


"""
From the Intern:
Searches through all the results that the CQM has chosen and outputs the characteristics
of the solution chosen for each leg, and the overall solution.
"""


def format_cqm_results(df, results):
    """
    Formats the CQM results into 2d-array (for easily parsing to CSV etc).
    Returns: list of lists

    df : pandas.DataFrame
    results : dimod.SampleSet
        filtered CQM results, as returned by `cqm_sample`
    """
    
    for sample in results.record:  
        counter=0
        modes_chosen=[]
        index_List=[]
        sol_num_list=[]

        if sample not in modes_chosen:
            modes_chosen.append(sample)
            counter+=1
        
            for indexSearch in range(len(sample[0])):
                if sample[0][indexSearch] == 1:
                    index_List.append(indexSearch)
    print(index_List)

    print(modes_chosen,"\n")
    print("Number of solutions chosen:", counter,"\n")

    length=0
    time=0
    cost=0
    env_cost=0

    counter2=0
    print("The combined solution is: \n")
    for item in index_List:
        counter2+=1
        print("Mode",counter2,"chosen:",df["modes"][item]," for leg",df["leg"][item])
        length+= df["length"][item]

        cost+= df["cost"][item]
        cost=round(cost,2)
        new_cost="$"+str(cost)

        time+= df["time"][item]
        env_cost+= df["env_cost"][item]
    
    print("\n DETAILS: \n","This route is", round(length,3), "miles long.","The cost of this route is ",new_cost, ", it will take you",round(time,1),"minutes to complete, and it will release",env_cost,"lbs of CO2. Safe traveling!")
                
    formatted_results = []
    imp_column_names = df.columns[2:-5]
    header = list(df["leg"].unique()) + list(imp_column_names) + ["energy", "num_occ"]
    formatted_results.append(header)

    # if no feasible solutions were found,
    if results is None:
        return formatted_results
        formatted_results.append(modes_chosen)

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


def print_hamiltonian(df, objective_column,one_hot_column, CONSTRAINTS):
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
    one_hot = format_one_hot(df, one_hot_column)
    # Get constraint string
    ineq = format_constraints(df, CONSTRAINTS)

    # Format and print QUBO
    print(show_cqm(objective, [ineq]))

def resolve_cqm(df, objective_column, one_hot_column,CONSTRAINTS, label):
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
    define_one_hot(df, cqm_model, one_hot_column)

    # Set inequality constraints
    define_constraints(df, cqm_model, CONSTRAINTS)

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

    # Print problem to screen
    print_hamiltonian(df, objective_column,one_hot_column, CONSTRAINTS)

    # Create sampler and retrieve location of results
    result_loc = resolve_cqm(df, objective_column, one_hot_column, constraints, label)

    # Load saved results
    loaded_results = load_results(result_loc)


"""
From the Intern:
This is the __main__ function, which actually runs the entire CQM. The one-hot & regular constraints are
defined in this function, as well as the objective column.
"""


if __name__ == "__main__":

    INPUT_CSV = dataFile
    OBJECTIVE_COLUMN = "time"
    ONE_HOT_COLUMN = "leg"
    CONSTRAINTS = [{"col_name" : "env_cost", "operator" : "<=", "comparison_value" : 100}]
    CONSTRAINTS.append({"col_name" : "cost", "operator" : "<=", "comparison_value" : 100})

    # name to associate with DWave sampling run
    NAME = outputFileName

    # human-readable timestamp (in case of multiple runs)
    TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    solve_cqm(INPUT_CSV, OBJECTIVE_COLUMN, ONE_HOT_COLUMN, CONSTRAINTS, f"{NAME}-{TIMESTAMP}")
