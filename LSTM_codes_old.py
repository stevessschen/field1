import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline  
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from keras.models import model_from_json
from sklearn.externals import joblib

#global features_col_name
filename = 'finalized_model.sav'
#Function to reshape dataset as required by LSTM
def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

# function to generate labels
def gen_label(id_df, seq_length, seq_cols,label):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    #plt.show()
    st.pyplot()

def prob_failure(machine_id, df_test, features_col_name, params):
    seq_length=params['2.1 Tune parameter: Window Size (min, hr, day)']
    #st.write('seq_length:', seq_length)
    #seq_length=50 #window_size #50, temparary added here !!!
    seq_cols = features_col_name #added!
    machine_df=df_test[df_test.id==machine_id]
    machine_test=gen_sequence(machine_df,seq_length,seq_cols)
    print(machine_test.shape)
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    
    m_pred=loaded_model.predict(machine_test)
    #print(m_pred)
    failure_prob=list(m_pred[-1]*100)[0]
    return failure_prob.round(4)
    
#Predicting...
def predict_data(dataset_predict, params):
    st.write("Data for prediction:",dataset_predict)
    X = dataset_predict.values[:,:]
    st.write('Shape of dataset:', X.shape, '=> ', X.shape[0], 'rows and ', X.shape[1], 'columns of dataset')

    features_col_name = list(dataset_predict.columns)
    features_col_name.remove('id')
    features_col_name.remove('cycle')
    #st.write("features_col_name:",features_col_name)
         
    #sc=MinMaxScaler() #Bug here !!!    
    scaler_filename = "saved_scaler"    
    sc = joblib.load(scaler_filename)
    dataset_predict[features_col_name] = sc.transform(dataset_predict[features_col_name])
    print('Shape of Test dataset: ',dataset_predict.shape)
    
    #machine_id=1
    machine_id = dataset_predict['id'].unique()
    machine_id2 = int(machine_id[0])
    #st.write("machine_id:",machine_id)
    period=params['2.2 Tune parameter: Cycle (min, hr, day)']
    st.write('Prediction: Probability that machine will fail within', period, 'Cycle (min, hr, day): ',prob_failure(machine_id2, dataset_predict, features_col_name, params), '%')
    
#Training...    
def show_data(dataset_train, classifier_name, params):
    st.write("Training dataset:",dataset_train) 
    X = dataset_train.values[:,:]
    st.write('Shape of dataset:', X.shape, '=> ', X.shape[0], 'rows and ', X.shape[1], 'columns of dataset')    
    st.write(f'Classifier = {classifier_name}', '=> model to train the dataset') 
    
    dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']
    
    df_train=dataset_train.copy()
    #period=30 #cycle #30
    period=params['2.2 Tune parameter: Cycle (min, hr, day)']
    #st.write("period:",period)
    df_train['label_bc'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
    #df_train.head()
    
    features_col_name = list(df_train.columns)
    print(features_col_name)
    features_col_name.remove('id')
    features_col_name.remove('cycle')
    features_col_name.remove('ttf')
    features_col_name.remove('label_bc')
    #print(features_col_name)
    
    target_col_name='label_bc'
    
    #Feature Scaling
    sc=MinMaxScaler()
    #state.my_value = "my value"
    df_train[features_col_name]=sc.fit_transform(df_train[features_col_name])
    scaler_filename = "saved_scaler"
    joblib.dump(sc, scaler_filename)

    # timestamp or window size
    seq_length=params['2.1 Tune parameter: Window Size (min, hr, day)']    
    #seq_length=50 #window_size #50
    #st.write("seq_length:",seq_length)
    seq_cols=features_col_name
    
    # generate X_train
    X_train=np.concatenate(list(list(gen_sequence(df_train[df_train['id']==id], seq_length, seq_cols)) for id in df_train['id'].unique()))
    #print(X_train.shape)
    # generate y_train
    y_train=np.concatenate(list(list(gen_label(df_train[df_train['id']==id], 50, seq_cols,'label_bc')) for id in df_train['id'].unique()))
    #print(y_train.shape)
    
    #LSTM Network
    nb_features =X_train.shape[2]
    timestamp=seq_length

    model = Sequential()

    model.add(LSTM(
         input_shape=(timestamp, nb_features),
         units=100,
         return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
          units=50,
          return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=5678) 
    # fit the network
    model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.05, verbose=1, callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    
    # training metrics
    scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)
    print('Accurracy: {}'.format(scores[1]))
    st.write("Accuracy of model on training data:", scores[1]) 
    
    #pickle.dump(model, open(filename, 'wb')) #not working!!!
    #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
    y_pred=model.predict_classes(X_test)
    st.write('Accuracy of model on test data: ',accuracy_score(y_test,y_pred))
    #print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
    
    #print(precision_score(y_test, y_pred, average=None).round(2))
    #print(recall_score(y_test, y_pred, average=None).round(2))
    Precision_lable = 'Precision: (0: normal 1: failure)'
    st.write(Precision_lable, precision_score(y_test, y_pred, average=None).round(2))
    Recall_lable = 'Recall: (0: normal 1: failure)'    
    st.write(Recall_lable, recall_score(y_test, y_pred, average=None).round(2))
    cm_dtc=confusion_matrix(y_test, y_pred)
    Confusion_matrix_lable = 'Confusion matrix: (0: normal 1: failure)'      
    st.write(Confusion_matrix_lable, cm_dtc)
    
    #https://stackabuse.com/understanding-roc-curves-with-python/
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    #Step 6: Predict probabilities for the test data.
    probs = model.predict_proba(X_test)
    #Step 7: Keep Probabilities of the positive class only.
    probs = probs[:, 0]
    #Step 8: Compute the AUC Score.
    auc = roc_auc_score(y_test, probs)
    #print('AUC: %.4f' % auc)
    st.write('AUC: %.4f' % auc)
    #Step 9: Get the ROC Curve.
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plot_roc_curve(fpr, tpr)
