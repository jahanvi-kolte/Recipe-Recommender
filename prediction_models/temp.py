from rec_engine import cdae_rec_engine as cdae
from utils import data_utils as du


#Configurable Parameters
MODEL_DIR_PATH="../trained_models"
MODEL_NAME="rl-100k_CDAE__200___1525136261.808919.h5"
PREDICT_DATASET="rl-100k-test"


#Fixed Parameters
MODEL_PATH=MODEL_DIR_PATH+"/"+MODEL_NAME
NUMBER_OF_ITEMS=1

#Get user data from query
#Only user id, list of ing id passed

#generating utility matrix
utility_matrix = du.load_utility_matrix(PREDICT_DATASET)
print(utility_matrix.shape)
utility_vector=du.vectorize(utility_matrix)



#Load the model
cdae_model_obj = cdae.CDAE(NUMBER_OF_ITEMS)
cdae_model_obj.load_model(MODEL_PATH)

#Predict user preference
prediction = cdae_model_obj.recommend_by_user_likes(utility_vector, 10)
print(prediction[0])
print(cdae_model_obj.model.summary())
#Filter based on recipes