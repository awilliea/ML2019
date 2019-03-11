import numpy as np
import pandas as pd
import sys

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
def output_result(filename,predict_value):
    id_ = []
    for i in range(predict_value.shape[0]):
        id_.append(str(i+1))
    output = pd.DataFrame(columns=['id','label'])
    output['id'] = id_
    output['label'] = predict_value
    output.to_csv(filename,index = False)
    
def feature_scaling(train_x,test_x):
    data = np.vstack([train_x,test_x])
    feature_mean = np.mean(data,axis = 0) 
    feature_var = np.std(data,axis = 0)
    data = (data - feature_mean)/feature_var 
    
    return data[:train_x.shape[0],:],data[train_x.shape[0]:,:]
def sigmoid(x):
    ans=1/(1+np.exp(-x))
    return np.clip(ans, 1e-8, 1 - (1e-8))

if __name__ == '__main__':
    train_file = sys.argv[3]
    test_file = sys.argv[5]
    output_file = sys.argv[6]
    
    features_X, train_fea_X = get_preprocessed_data(train_file)
    features_X_test, test_fea_X = get_preprocessed_data(test_file)
    train_fea_X,test_fea_X = feature_scaling(train_fea_X,test_fea_X)
    
    test_x = np.hstack([test_x,np.ones((test_x.shape[0],1))])
    opt_w = np.array([  2.99283737e-01,   6.42907988e-02,   3.03893415e-01,
         1.39239162e+00,   2.28629633e-01,   3.17384835e-01,
         8.98436437e-02,  -1.74326561e-02,   3.02137988e-04,
         4.82365405e-02,   5.43742311e-02,  -8.01836592e-02,
        -2.93989643e-02,  -4.32722611e-02,  -6.34679788e-02,
        -1.37984964e-01,  -1.56408434e-01,  -6.27583469e-02,
        -8.68158471e-02,  -1.27887108e-01,  -1.81964753e-01,
        -1.53605462e-01,   2.46392364e-02,   3.14724923e-02,
         2.60495240e-01,   1.75013324e-01,  -1.54594230e-01,
         2.39319820e-01,  -1.21411166e-01,   1.93875660e-01,
        -1.32947862e-02,  -1.09853062e-01,   3.15611542e-02,
         4.03606345e-01,  -4.19042252e-02,  -3.03880224e-01,
        -6.09438725e-02,  -3.58625962e-02,  -2.66768784e-02,
        -1.78869953e-02,   6.11364526e-03,   2.23192651e-01,
        -1.58180960e-01,  -1.23641147e-01,  -8.46374118e-02,
        -2.07122830e-01,  -1.02901164e-01,   1.52546800e-01,
         5.91702221e-02,   6.57909462e-02,   9.58308300e-02,
        -2.85909886e-02,  -6.33378326e-02,   1.75895539e-01,
        -4.89021345e-02,  -9.21024154e-02,  -2.49614033e-01,
        -8.06872565e-02,   3.09440754e-01,  -2.97166462e-02,
         1.41659199e-02,  -1.58517359e-02,  -3.83247026e-02,
         2.42760966e-02,   1.19327342e-02,   2.56015972e-02,
        -3.05218047e-02,  -5.86954149e-02,   1.25733209e-02,
        -5.27753233e-02,  -2.90387536e-02,  -2.74792831e-02,
         1.93648919e-02,   1.68211098e-02,   3.96975459e-02,
        -2.46912138e-02,  -4.09656116e-02,  -4.49281799e-03,
        -6.82928876e-03,  -9.77517656e-03,   5.02871156e-03,
        -1.29013821e-02,  -1.50192122e-02,  -1.05362576e-03,
         2.55508573e-02,   4.37336091e-02,   1.82018780e-02,
         2.27661932e-02,  -1.79516349e-02,  -6.00453999e-02,
        -1.77421831e-02,  -3.21548933e-02,  -3.07690978e-02,
         2.44175876e-02,   2.43439815e-03,   4.30039764e-03,
        -8.21415305e-03,  -1.02626845e-02,  -3.33473130e-02,
         3.47351256e-03,  -1.01637710e-02,  -1.66750522e-02,
         6.10760315e-02,  -5.84111085e-02,   1.98819382e-02,
        -2.70316439e-02,  -1.80224192e+00])
    
    temp_test = sigmoid(np.dot(test_fea_X,opt_w))
    pre_test = np.array([float(x>0.5) for x in temp_test]).astype(np.int)
    output_result(output_file,pre_test)