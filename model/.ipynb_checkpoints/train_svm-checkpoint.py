# !pip install scikit-multilearn 

from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
from sklearn.externals import joblib
# Import joblib package directly
#import joblib

## TODO: Import any additional libraries you need to define a model
#from sklearn.linear_model import LogisticRegression

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model

# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])




if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
#     install('scikit-multilearn')
#     from skmultilearn.problem_transform import BinaryRelevance
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    print('SM_TRAIN is ',os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    print(training_dir)
    train_data = pd.read_csv(os.path.join(training_dir, "train-processed.csv"))

    # comments are in first column and rest are labels
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_x = train_data['comment_text']
    train_y = train_data[labels]
    

    ## Define a model 
    classifier = OneVsRestClassifier(SVC(gamma='scale', class_weight='balanced'), n_jobs=1)
    
    model = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        strip_accents='unicode',     
                        analyzer='word',            
                        token_pattern=r'\w{1,}',    
                        ngram_range=(1, 3),         
                        stop_words='english',
                        sublinear_tf=True)
                    ),
                    ('clf', classifier)
                ])
    
    
    ## Train the model
    model.fit(train_x, train_y)
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))