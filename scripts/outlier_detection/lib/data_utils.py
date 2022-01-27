# libraries
import pandas as pd
from typing import Any, Union, Dict
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame
import collections


# functions
def convert_to_date(unix_time):
    result_ms = pd.to_datetime(unix_time, unit='ms')
    str(result_ms)
    return result_ms


def is_columns_same(df, column1, column2):
    is_same = df[column1].equals(df[column2])
    return is_same


def count_string_frequency(input_all_strings):
    str_count: Dict[Any, int] = dict()
    for str in input_all_strings:
        if str in str_count:
            str_count[str] += 1
        else:
            str_count[str] = 1

    for key, value in str_count.items():
        print("% s : % d" % (key, value))

    return str_count


def n_most_common_in_series(series, n):
    d = collections.Counter(series)
    n_most_common = d.most_common(n)
    return n_most_common
