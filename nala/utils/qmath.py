import math


# todo implement test functions
def mean(arr_of_nr):
    return sum(arr_of_nr)/len(arr_of_nr)

def hmean(arr_of_pos_nr):
    return len(arr_of_pos_nr) / sum(1/x for x in arr_of_pos_nr)
