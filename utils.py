import pickle
import os

def pickle_data(object_to_pickle, outfile_path):
    object = []
    if os.path.exists(outfile_path):
        with open(outfile_path,"rb") as file:
            # object_list = pickle.load(file)
            # object.extend(object_list)
            object = pickle.load(file)
    object.append(object_to_pickle)

    with open(outfile_path,"wb") as file:
        pickle.dump(object,file)
    
    # with open(outfile_path,"rb") as file:
    #     print(pickle.load(file))




