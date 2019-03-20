import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

def output_result(filename,predict_value):
    id_ = []
    for i in range(predict_value.shape[0]):
        id_.append(str(i+1))
    output = pd.DataFrame(columns=['id','label'])
    output['id'] = id_
    output['label'] = predict_value
    output.to_csv(filename,index = False)
    
def get_preprocessed_data(filename):
    train_fea = []
    i = 0
    with open(filename) as f:
        for line in f.readlines():
            if(i == 0):
                features = line.split(',')
                i = 1
            else:
                train_fea.append(line.split(','))
    return np.array(features), np.array(train_fea).astype(np.float64)

if __name__ == '__main__':
    train_file = sys.argv[3]
    test_file = sys.argv[5]
    output_file = sys.argv[6]
    
    features_X_test, test_fea_X = get_preprocessed_data(test_file)
    gbm2 = joblib.load('gbm2.pkl')
    pre_test = gbm2.predict(test_fea_X).astype(int)
    output_result(output_file,pre_test)
   