from rec_engine import cdae_rec_engine as cdae
import time
import os.path,config,json
from utils import data_utils as du
from utils import process_utils as pu


def train_new_cdae_model():
    # Configurable Params
    DATASET = "rl-45km-base"
    CDAE_MODEL_NAME = "rl-45km"
    BATCH_SIZE = 500
    EPOCHS = 20

    with open(os.path.join(config.__path__[0], 'rr_config.json')) as json_file:
        config_details = json.load(json_file)
        MODEL_LABEL = config_details["model-name"]

        # Fixed Params
        TRAINED_MODELS_PATH = "../trained_models"

        # It picks up the correct configuration according to utility matrix
        NUMBER_OF_ITEMS = 1

        # generating utility matrix
        utility_matrix = du.load_utility_matrix(DATASET)
        print(utility_matrix.shape)
        utility_vector = du.vectorize(utility_matrix)

        # creating model object for CDAE
        cdae_model_obj = cdae.CDAE(NUMBER_OF_ITEMS)

        start_time = time.time()

        # Train CDAE Model
        cdae_model_obj.train_model(utility_vector, BATCH_SIZE, EPOCHS)
        cdae_model_obj._make_predict_function()
        end_time = time.time()

        # Print CDAE Model Summary
        print("Model training time : {}", end_time - start_time)

        #backup_ml_model_file
        pu.backup_ml_model_file()

        # Store CDAE Model
        cdae_model_obj.save_model(TRAINED_MODELS_PATH, MODEL_LABEL)

        return cdae_model_obj