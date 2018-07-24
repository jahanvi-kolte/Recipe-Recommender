import config
import numpy
import json
import os
from keras.utils import to_categorical


def load_utility_matrix(dataset ='rl-45km-base'):
    #remove 2nd and 3rd parameter later

    # Open config file
    with open(os.path.join(config.__path__[0], 'rr_config.json')) as json_file:
        config_details = json.load(json_file)

        filename = config_details[dataset]
        dataset = dataset.rstrip('-base')
        data_folder = os.path.join(config_details["dataFolder"], dataset)

        print("Loading Data from {}".format(data_folder))

        max_item_id=-1
        user_item_map={}

        try:
            data_file = os.path.join(data_folder, filename)
        except KeyError:
            print("Exception KeyError: Tried to access invalid key in config file")
            return None

        print("Opening datafile ",data_file)
        with open(data_file, 'r') as file:
            next(file)
            for line in file:
                user_id, item_id, liked, timestamp = line.rstrip().split(',')


                if max_item_id < int(item_id):
                    max_item_id = int(item_id)

                if int(user_id) not in user_item_map:
                    user_item_map[int(user_id)] = [int(item_id)]
                else:
                    user_item_map[int(user_id)].append(int(item_id))


            max_item_id = max_item_id+1
    return numpy.array([numpy.sum(to_categorical(items, max_item_id), axis=0) for items in user_item_map.values()])


def create_utility_matrix_input(user_like_history):
    with open(os.path.join(config.__path__[0], 'rr_config.json')) as json_file:
        config_details = json.load(json_file)
        #json = json.loads(json)

        max_count = int(config_details["recipe_count"])

        return numpy.array([numpy.sum(to_categorical(user_like_history["recipeId"], max_count), axis=0)])


def vectorize(utility_matrix):
    return utility_matrix.flatten();

if __name__ == "__main__":
    #result = load_utility_matrix()
    data='{"userId": 1, "recipeId": [1,0,2, 7, 89, 124]}'
    json_data = json.loads(data)

    #print(json)
    #print(json["recipeId"])
    result = create_utility_matrix_input(json_data)
    print(result)

