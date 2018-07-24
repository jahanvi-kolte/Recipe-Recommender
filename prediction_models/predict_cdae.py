from utils import data_utils as du
import json
from rec_engine import cdae_rec_engine as cdae



def get_recipe_rankings(model,json_query):
    input_um = du.create_utility_matrix_input(json_query)
    utility_vector = du.vectorize(input_um)
    prediction = model.recommend_by_user_likes(utility_vector, 500)
    return prediction[0]


if __name__ == "__main__":
    MODEL_DIR_PATH = "../trained_models"
    MODEL_NAME = "rl-100k_CDAE__200___1525136261.808919.h5"

    # Fixed Parameters
    MODEL_PATH = MODEL_DIR_PATH + "/" + MODEL_NAME
    NUMBER_OF_ITEMS = 1

    data = '{"userId": 1, "recipeId": [1,0,2, 7, 89, 124]}'
    json_data = json.loads(data)

    # Load the model
    cdae_model_obj = cdae.CDAE(NUMBER_OF_ITEMS)
    cdae_model_obj.load_model(MODEL_PATH)
    result=get_recipe_rankings(cdae_model_obj,json_data)
    print(result)