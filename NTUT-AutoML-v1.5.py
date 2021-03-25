#Version 1.0 copy codes from youtube.com, changed to NTUT autoML
#Version 1.5 Added KMeans and autocluster

import streamlit as st 
import os #Error: OpenBLAS blas_thread_init: pthread_create: Resource temporarily unavailable
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np 
import datetime
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import pickle
import LSTM_codes
import autoML_Reg_codes
import Cluster_KMeans_codes

st.set_option('deprecation.showfileUploaderEncoding', False)
#st.set_option('deprecation.showPyplotGlobalUse', False)
global X_test, y_test
st.title('NTUT AutoML')
st.write('Automatic Machine Learning designed by NTUT Industry 4.0 Consulting Group (v1.5)')
st.write('(Read Quick Testing Guide and follow 5 steps on the left panel to train a model and predict!)')
#st.write("""
# Explore different classifier and datasets
#Which one is the best?
#""")
st.cache(allow_output_mutation=True)
def get_dataset(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data
    #data = None
    #if name == 'Iris':
    #    data = datasets.load_iris()
    #elif name == 'Wine':
    #    data = datasets.load_wine()
    #else:
    #    data = datasets.load_breast_cancer()
    #X = data.data
    #y = data.target
    #st.write(X, y)
    #X = df.values[:,1:14]
    #y = df.values[:,0]
    #return X, y

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('2.1 Tune parameter: C', 0.01, 10.0, 0.15)
        params['2.1 Tune parameter: C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('2.1 Tune parameter: K', 1, 15)
        params['2.1 Tune parameter: K'] = K
    elif clf_name == 'Cluster: KMeans':
        no_cluster = st.sidebar.slider('2.1 Tune parameter: No of Clusters', 1, 15, 4)
        params['2.1 Tune parameter: No of Clusters'] = no_cluster
        center_seed = st.sidebar.slider('2.2 Tune parameter: initial cluster center seed', 1, 15, 5)
        params['2.2 Tune parameter: initial cluster center seed'] = center_seed
    elif clf_name == 'autoML':
        Time = st.sidebar.slider('2.1 Tune parameter: Time (Second)', 1, 120, 30)
        params['2.1 Tune parameter: Time (Second)'] = Time   
    elif clf_name == 'autoML: Regression':
        Generation = st.sidebar.slider('2.1 Tune parameter: Generation (Epoch)', 1, 100, 1)
        params['2.1 Tune parameter: Generation (Epoch)'] = Generation    
    elif clf_name == 'LSTM-Time Series':
        window_size = st.sidebar.slider('2.1 Tune parameter: Window Size (min, hr, day)', 12, 120, 50)
        params['2.1 Tune parameter: Window Size (min, hr, day)'] = window_size 
        cycle = st.sidebar.slider('2.2 Tune parameter: Cycle (min, hr, day)', 12, 120, 30)
        params['2.2 Tune parameter: Cycle (min, hr, day)'] = cycle
    else:
        max_depth = st.sidebar.slider('2.1 Tune parameter: max_depth', 2, 15, 5)
        params['2.1 Tune parameter: max_depth'] = max_depth
        n_estimators = st.sidebar.slider('2.2 Tune parameter: n_estimators', 1, 100, 15)
        params['2.2 Tune parameter: n_estimators'] = n_estimators
    return params

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['2.1 Tune parameter: C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['2.1 Tune parameter: K'])
    #elif clf_name == 'autoML':
        #clf = KNeighborsClassifier(Time=params['2.1 Tune parameter: Time (Second)'])        
    else:
        clf = RandomForestClassifier(n_estimators=params['2.2 Tune parameter: n_estimators'], 
            max_depth=params['2.1 Tune parameter: max_depth'], random_state=1234)
    return clf
 
def autoML(X, y):
    #import sklearn.datasets
    import sklearn.metrics
    import autosklearn.classification
    global y_test
    global y_pred
    #X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)
    #st.write('Shape of dataset:', X.shape)
    #st.write('Number of classes:', len(np.unique(y)))
    Time=params['2.1 Tune parameter: Time (Second)']
    current_date_and_time = datetime.datetime.now()
    current_date_and_time_string = str(current_date_and_time)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=Time, #30, 120
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_classification_example_tmp_'+current_date_and_time_string,
        output_folder='/tmp/autosklearn_classification_example_out_'+current_date_and_time_string,
        )
    #automl.fit(X_train, y_train, dataset_name='breast_cancer')
    automl.fit(X_train, y_train)    
    st.write('Show model:', automl.show_models())
    y_pred = automl.predict(X_test)
    #print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
    #st.write(f'Classifier = {classifier_name}')
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    st.write(f'Accuracy =', acc.round(2))
    # save the model to disk
    model=automl
    pickle.dump(model, open(filename, 'wb'))    
    return model, X_test, y_test
    
uploaded_file = st.sidebar.file_uploader("1. Upload training dataset", type="csv")
classifier_name = st.sidebar.selectbox(
    '2. Select classifier',
    ('KNN', 'SVM', 'Random Forest', 'Cluster: KMeans', 'autoML: autocluster(pending...)', 'autoML', 'autoML: Regression', 'LSTM-Time Series')
)
#if classifier_name != 'autoML':
params = add_parameter_ui(classifier_name)
submit = st.sidebar.button('3. Train!')
uploaded_file2 = st.sidebar.file_uploader("4. Upload dataset to predict", type="csv")
submit2 = st.sidebar.button('5. Predict')
filename = 'finalized_model.sav'
def y_to_number(data, list_y):
    y = data.values[:,0]   
    for i in range(0,len(np.unique(y))):
        data[data.columns[0]].replace({list_y[i]: i}, inplace=True)        
        #data.columns[0].replace({'White Wine': i}, inplace=True)
    return data
def number_to_y(y, list_y, y_pred):
    for i in range(0,len(np.unique(y))):
        #y_pred[y_pred == i] = list_y[i]
        #y_pred = np.where(y_pred==i, list_y[i], y_pred) 
        #y_pred[y_pred.columns[0]].replace({i: list_y[i]}, inplace=True)  
        y_pred = pd.DataFrame(data=y_pred)
        y_pred[y_pred.columns[0]].replace({i: list_y[i]}, inplace=True)
        y_pred = np.array(y_pred)
    return y_pred

if submit2 and classifier_name == 'LSTM-Time Series': #Predict Buttom + LSTM
    #data2 = pd.read_csv(uploaded_file) #Training data
    data = pd.read_csv(uploaded_file2) #Predict data    
    LSTM_codes.predict_data(data, params)
if submit2 and classifier_name == 'autoML: Regression': #Predict Buttom + autoML: Regression
    #data2 = pd.read_csv(uploaded_file) #Training data
    data = pd.read_csv(uploaded_file2) #Predict data    
    autoML_Reg_codes.predict_data(data, params)
if submit2 and classifier_name == 'Cluster: KMeans': #Predict Buttom + Cluster: KMeans
    #data2 = pd.read_csv(uploaded_file) #Training data
    data = pd.read_csv(uploaded_file2) #Predict data    
    Cluster_KMeans_codes.predict_data(data, params)     
if submit2 and classifier_name != 'LSTM-Time Series' and classifier_name != 'autoML: Regression' and classifier_name != 'Cluster: KMeans' : #Predict buttom
    data2 = pd.read_csv(uploaded_file) #Training data
    data = pd.read_csv(uploaded_file2) #Predict data
    y = data2.values[:,0]    
    st.write("Data before prediction:",data)
    data_input = data.values[:,1:]
    data_input = data_input.astype('float64') 
    data_output = data
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(data_input)
    list_y = np.unique(y) #ng,ok
    #list_y
    #st.write(len(np.unique(y)))
    y_pred2 = number_to_y(y, list_y, y_pred)
    data_output[data.columns[0]].replace({'?': y_pred2}, inplace=True)
    st.write("Data after prediction:",data_output)    

if submit and classifier_name == 'LSTM-Time Series': #Train Buttom + LSTM
    data3 = get_dataset(uploaded_file)
    LSTM_codes.show_data(data3, classifier_name, params)
if submit and classifier_name == 'autoML: Regression': #Train Buttom + autoML: Regression
    data3 = get_dataset(uploaded_file)
    autoML_Reg_codes.show_data(data3, classifier_name, params)
if submit and classifier_name == 'Cluster: KMeans': #Train Buttom + Cluster: KMeans
    data3 = get_dataset(uploaded_file)
    Cluster_KMeans_codes.show_data(data3, classifier_name, params)    

    
if submit and classifier_name != 'LSTM-Time Series' and classifier_name != 'autoML: Regression' and classifier_name != 'Cluster: KMeans': #Train Buttom
    data3 = get_dataset(uploaded_file)        
    data  = data3.copy()
    data2 = data3.copy()
    st.write("Training dataset:",data2)    
    y = data.values[:,0]   
    
    list_y = np.unique(y) #ng,ok
    #st.write("list_y:",list_y)       
    data = y_to_number(data, list_y) #handle label string:ok,ng => convert it to 0,1
    #st.write("Training dataset convert to number:",data)     
    
    X = data.values[:,1:]
    y = data.values[:,0]      
    st.write('Shape of dataset:', X.shape, '=> ', X.shape[0], 'rows and ', X.shape[1], 'columns of dataset')    
    st.write('Number of classes:', len(np.unique(y)), '=> ', len(np.unique(y)), 'categories of dataset')
    st.write(f'Classifier = {classifier_name}', '=> model to train the dataset')    
    
    if classifier_name == 'autoML':
        #model, X_test, y_test = autoML(X, y)
        #pass
        clf = RandomForestClassifier(n_estimators=5, max_depth=15, random_state=1234)
        #### CLASSIFICATION ####
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5678) 
        clf.fit(X_train, y_train)
        # save the model to disk
        model=clf
        pickle.dump(model, open(filename, 'wb'))    
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy =', acc.round(2))        
    else: #Steve: comment out!
        #params = add_parameter_ui(classifier_name)
        clf = get_classifier(classifier_name, params)
        #### CLASSIFICATION ####
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5678) 
        clf.fit(X_train, y_train)
        # save the model to disk
        model=clf
        pickle.dump(model, open(filename, 'wb'))    
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy =', acc.round(2))
        #Steve: else to here!
    
    #st.write(list_y)
    lable_1 = ''
    for i in range(0,len(np.unique(list_y))):
        lable_1 = lable_1 + str(i) + ': ' + str(list_y[i])
        if i < len(np.unique(list_y)) - 1:
            lable_1 = lable_1 + ', '
    Precision_lable = 'Precision: (' + lable_1 + ')'
    st.write(Precision_lable, precision_score(y_test, y_pred, average=None).round(2))
    Recall_lable = 'Recall: (' + lable_1 + ')'    
    st.write(Recall_lable, recall_score(y_test, y_pred, average=None).round(2))
    cm_dtc=confusion_matrix(y_test, y_pred)
    Confusion_matrix_lable = 'Confusion matrix: (' + lable_1 + ')'      
    st.write(Confusion_matrix_lable, cm_dtc)
    
    st.write('Graph for reference ONLY: ')
    st.write('Project the data onto the 2 primary principal components: ')    
    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig = plt.figure()
    plt.scatter(x1, x2,
    c=y, alpha=0.8,
    cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    #plt.show()
    st.pyplot(fig)        
    
    #if len(np.unique(y)) == 2:
    #    st.write('ROC Curve: ', y_test, model)    
    #    plot_roc_curve(model, X_test, y_test)
    #    st.pyplot()
