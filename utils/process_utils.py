import os,config,json
import time

def backup_ml_model_file():
    with open(os.path.join(config.__path__[0], 'rr_config.json')) as json_file:
        config_details = json.load(json_file)

    model_fname = config_details["model-name"]
    backup_model_fname=model_fname+"_"+time.strftime("%Y%m%d-%H%M%S")

    os.rename(model_fname,backup_model_fname)

def load_backup_ml_model():
    with open(os.path.join(config.__path__[0], 'rr_config.json')) as json_file:
        config_details = json.load(json_file)

        #rename
        #return ml model


