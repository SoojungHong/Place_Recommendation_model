# libraries
import pandas as pd


# functions
def read_file_without_header(file):
    data = pd.read_csv(file, delimiter='\t')
    return data


def read_file(file, header):
    data = pd.read_csv(file, sep='\t', names=header, error_bad_lines=False)
    return data


def read_data_add_factorized_id(file, header):
    data = read_file(file, header)
    data['user_int'] = pd.factorize(data.cookie)[0]
    data['place_int'] = pd.factorize(data.result_name)[0]
    return data
