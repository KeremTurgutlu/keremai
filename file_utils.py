import pickle

def write_pickle(target_file, obj):
    """
    Pickle a python object to target file
    """
    with open(target_file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(source_file):
    """
    Read a pickled python object from source file
    """
    with open(source_file, 'rb') as handle:
        obj = pickle.load(handle)
    return obj