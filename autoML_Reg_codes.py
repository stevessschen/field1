import streamlit as st 
from tpot import TPOTRegressor #https://github.com/EpistasisLab/tpot
from sklearn.model_selection import train_test_split
import pickle
import sys
#from sklearn.externals import joblib

#global filename
filename = 'finalized_model_autoML_Reg.sav'

#Predicting...
def predict_data(dataset_predict, params):
    #data2 = pd.read_csv(uploaded_file) #Training data
    #data = pd.read_csv(uploaded_file2) #Predict data
    #y = data2.values[:,0]
    data = dataset_predict
    st.write("Data before prediction:",data)
    data_input = data.values[:,1:]
    #data_input = data_input.astype('float64') #Comment out! trying!!
    data_output = data
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(data_input)
    #list_y = np.unique(y) #ng,ok
    #list_y
    #st.write(len(np.unique(y)))
    #y_pred2 = number_to_y(y, list_y, y_pred)
    #data_output[data.columns[0]].replace({'?': y_pred2}, inplace=True)
    data_output[data.columns[0]].replace({'?': y_pred}, inplace=True)
    st.write("Data after prediction:",data_output)    

#Training...    
def show_data(dataset_train, classifier_name, params):
    st.write("Training dataset:",dataset_train) 
    X = dataset_train.values[:,1:]
    y = dataset_train.values[:,0]
    st.write('Shape of dataset:', X.shape, '=> ', X.shape[0], 'rows and ', X.shape[1], 'columns of dataset')    
    st.write(f'Classifier = {classifier_name}', '=> model to train the dataset') 

    generation=params['2.1 Tune parameter: Generation (Epoch)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

    tpot = TPOTRegressor(generations=generation, population_size=50, verbosity=2, random_state=42) #generations=5
    tpot.fit(X_train, y_train)
    #st.write('Info for reference only:', tpot.fit(X_train, y_train))
    #print(tpot.score(X_test, y_test))

    tpot.export('tpot_boston_pipeline.py')
    #tpot.log('tpot_progress_content.txt')
    MSE = abs(tpot.score(X_test, y_test))
    st.write("MSE (Mean Squared Error):", MSE.round(2))
    
    #st.write(tpot.evaluated_individuals_)
    
    # save the model to disk
    #model=tpot
    #pickle.dump(model, open(filename, 'wb')) 
    
    #from joblib import dump, load
    #dump(tpot, 'filename.joblib') 
    
    #https://github.com/EpistasisLab/tpot/issues/11#issuecomment-341421022
    pickle.dump(tpot.fitted_pipeline_, open(filename, 'wb'))
