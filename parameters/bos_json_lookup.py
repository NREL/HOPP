import json


def bos_json_lookup(json_file_path, search_matrix, desired_output_parameter, verbosity=1):
    # Loads the json data containing all the BOS cost information from the excel model
    # If the project parameters defined in "search matrix" are present in the table, a summary of the record or records
    # is printed, and the desired_output_parameter is returned
    with open(json_file_path, "r") as read_file:
        'Load the json data'
        data = json.load(read_file)

        # For each run in the json data
        # print the desired_output_parameter
        # for entries which match the values in search_matrix (if they exist)'
        for run_record in data:
            match = search_bos_model_run_record(run_record, search_matrix, desired_output_parameter)
            if match is not None and verbosity >= 1:
                print(match)
    # return the desired_output_parameter
    return match


def search_bos_model_run_record(run_record, search_matrix, desired_output_parameter, verbosity=1):
    if run_record["Scenario Type:"] == search_matrix[0] and run_record["Project Capacity Solar"] <= search_matrix[1] and \
            run_record["Project Capacity Wind"] == search_matrix[2]:
        if verbosity >= 1:
            print("For project scenario: '" + search_matrix[0] + "', with " + str(run_record["Project Capacity Solar"]) \
                  + "MW of installed Solar Capacity of type: '" + run_record["Project Type"] + "', and " \
                  + str(run_record["Project Capacity Wind"]) + "MW of installed Wind Capacity")
            print("The Total Project cost is: $" + str(run_record["Total Project Cost"]))

        # return the desired_output_parameter for the run_record matching the parameters in search_matrix
        return run_record[desired_output_parameter]


def bos_json_lookup_custom(json_file_path, search_matrix, desired_output_parameters, verbosity=0):
    # Loads the json data containing all the BOS cost information from the Excel model
    # If the project parameters defined in "search matrix" are present in the table, a summary of the record or records
    # which match are printed, and the desired_output_parameter(s) are returned.
    with open(json_file_path, "r") as read_file:
        json_data = json.load(read_file)
        match_list = []
        # For each run in the json data
        # print and return the desired_output_parameters
        # for entries which match the values in search_matrix (if they exist)
        # Check if user used an integer input to search_matrix
        if (any(type(item) == int for key, item in search_matrix.items())):
            print("Please specify search matrix inputs in list format. For single values, use e.g. [100]")
        else:
            for run_record in json_data:
                match = search_bos_model_run_record_custom(run_record, search_matrix, desired_output_parameters)

                if match is not None:
                    if verbosity >= 1:
                        print("Match Found")
                    match_list.append(match)
            # return the desired_output_parameter /  list of the desired parameter matches
            return match_list


def search_bos_model_run_record_custom(run_record, search_matrix, desired_output_parameters):
    # returns a dict of desired_output_parameter names and values if a run_record matches the search values defined
    # for all parameters in search_matrix

    if all(run_record[key] in value for key, value in search_matrix.items()): # For matching a range of items
        # return the desired_output_parameter for the run_record matching the parameters in search_matrix
        # print(list(run_record[d] for d in desired_output_parameters))
        return dict((d, run_record[d]) for d in desired_output_parameters)
