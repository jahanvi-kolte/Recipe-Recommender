from flask import Flask, jsonify, request
from rec_engine import cdae_rec_engine as cdae
from prediction_models import predict_cdae
import os
import config
import json
from training_models import train_cdae as train_cdae

app = Flask(__name__)

#app.run(host="0.0.0.0", port="5050")

with open(os.path.join(config.__path__[0], 'rr_config.json')) as json_file:
    config_details = json.load(json_file)


#Configurable Parameters
MODEL_DIR_PATH=config_details["model-dir-path"]
MODEL_NAME=config_details["model-name"]

#Fixed Parameters
MODEL_PATH=MODEL_DIR_PATH+"/"+MODEL_NAME
NUMBER_OF_ITEMS=1


#Load the model
cdae_model_obj = cdae.CDAE(NUMBER_OF_ITEMS)
cdae_model_obj.load_model(MODEL_PATH)


@app.route('/rankings', methods=['POST'])
def get_recipe_ranks():
    response = {}
    try:
        query=request.get_json();
        result = predict_cdae.get_recipe_rankings(cdae_model_obj,query)
        response["userId"]=query["userId"]
        response["rankedRecipeIds"]=result.tolist()
        response = jsonify(response)
        return response, 200
    except ValueError as e:
        print("Error in get_recipe_ranks()", e)
        return response,500



@app.route('/train', methods=['GET'])
def train_ml_model():
    #Model backup
    try:
        print("here")
        cdae_new_model = train_cdae()
    except ValueError as e:
        print("Error in train_ml_model() while training", e)

    try:
        cdae_model_obj=cdae_new_model
        return 200
    except ValueError as e:
        print("Error in train_ml_model() while model switching", e)
        return 500



