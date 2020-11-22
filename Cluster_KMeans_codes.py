import streamlit as st 
#KMeans (https://machinelearningmastery.com/clustering-algorithms-with-python/)
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from matplotlib import pyplot
import pickle
#from sklearn.externals import joblib

#global filename
filename = 'finalized_model_Cluster_KMeans.sav'

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
    #data_output[data.columns[0]].replace({'?': y_pred}, inplace=True)
    data_output.insert(loc=0, column='Cluster', value=y_pred)
    st.write("Data after prediction:",data_output)    

#Training...    
def show_data(dataset_train, classifier_name, params):
    st.write("Training dataset:",dataset_train) 
    X = dataset_train.values[:,1:]
    #y = dataset_train.values[:,0]
    st.write('Shape of dataset:', X.shape, '=> ', X.shape[0], 'rows and ', X.shape[1], 'columns of dataset')    
    st.write(f'Classifier = {classifier_name}', '=> model to train the dataset') 

    no_cluster=params['2.1 Tune parameter: No of Clusters']
    center_seed=params['2.2 Tune parameter: initial cluster center seed']
    # define the model
    model = KMeans(n_clusters=no_cluster, random_state=center_seed)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)                      
    #dataset_train['Cluster']=yhat
    dataset_train.insert(loc=0, column='Cluster', value=yhat)
    st.write("Resulting dataset:", dataset_train.sort_values(by=['Cluster'])) 
    
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    #pyplot.show()
    st.pyplot()    
    
    # save the model to disk
    pickle.dump(model, open(filename, 'wb')) 