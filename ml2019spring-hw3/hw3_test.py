from keras.models import load_model
import pandas as pd
import numpy as np

def preprocess_x(train):
    train_x = []
    for f in train['feature']:
        temp = np.array(f.split(' ')).astype(int)
        train_x.append(temp)
    
    train_x = np.array(train_x)
    train_x = train_x.reshape((train_x.shape[0],48,48,1))
    return train_x
def output_result(filename,predict_value):
    id_ = []
    for i in range(predict_value.shape[0]):
        id_.append(i)
    output = pd.DataFrame(columns=['id','label'])
    output['id'] = id_
    output['label'] = predict_value
    output.to_csv(filename,index = False)

if __name__ == '__main__':
    test_file = sys.argv[1]
    output_file = sys.argv[2]
    
    test = pd.read_csv(test_file,engine='python')
    test_x = preprocess_x(test)
    
    model = load_model('model_data_gen_5.h5')
    pre_test = model.predict_classes(test_x/255,batch_size=128)
    output_result(output_file,pre_test)
