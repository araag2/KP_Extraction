import pickle

# -----------------------------------------------------------------------------
# write_to_file() - Small auxiliary function to write data to a file
# -----------------------------------------------------------------------------
def write_to_file(filename, con):
    with open('{}.txt'.format(filename), 'wb') as write_f:
        pickle.dump(con, write_f)
    return

# -----------------------------------------------------------------------------
# read_from_file() - Small auxiliary function to read data from a file
# -----------------------------------------------------------------------------
def read_from_file(filename):
    with open('{}.txt'.format(filename), 'rb') as read_f:
        return pickle.load(read_f) 