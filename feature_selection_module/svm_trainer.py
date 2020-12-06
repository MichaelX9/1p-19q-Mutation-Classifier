#!/usr/bin/env python3
import numpy as np
import pandas as pd  
import scipy
import scipy.io
import sklearn 
import os
import pymrmr
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import FastICA
####FOR PYMRMR SELECTION SETTINGS: https://link.springer.com/chapter/10.1007/978-3-642-04180-8_47
dataPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/aboveslices/T2/corr_select'
testPath = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/aboveslices/T2'
train_file = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/aboveslices/T1/corr_select/pymrmr_select.csv'
test_file = '/mnt/c/Users/Michael/Desktop/PyRadiomics/TestSetFeatures/aboveslices/T1/pymrmr_select.csv'
#feature_file_3 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T1/below_slices/selected_.7.csv'
#feature_file_4 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T2/below_slices/selected_.7.csv'
#feature_file_5 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T1/center_slices/selected_.7.csv'
#feature_file_6 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T2/center_slices/selected_.7.csv'
feature_file_3 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/belowslices/T1/corr_select/selected_.7.csv'
feature_file_4 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/belowslices/T2/corr_select/selected_.7.csv'
feature_file_5 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/centerslices/T1/corr_select/selected_.7.csv'
feature_file_6 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/centerslices/T2/corr_select/selected_.7.csv'
#feature_file_6 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/slicewise-features/T2/center_slices/selected_.5.csv'
#feature_file_2 = '/mnt/c/Users/Michael/Desktop/PyRadiomics/GeneratedFeatures/aboveslices/T2/correlation_selection/selected_.7.csv'
deletion_nums = ['LGG-219', 'LGG-273', 'LGG-277', 'LGG-280', 'LGG-285', 'LGG-297', 'LGG-327', 'LGG-338', 'LGG-343', 'LGG-351', 'LGG-354', 'LGG-363', 'LGG-374', 'LGG-375', 'LGG-391', 'LGG-516', 'LGG-203', 'LGG-286', 'LGG-533', 'LGG-545', 'LGG-558', 'LGG-574', 'LGG-585', 'LGG-589', 'LGG-601', 'LGG-622', 'LGG-625', 'LGG-306', 'LGG-631', 'LGG-210', 'LGG-240', 'LGG-263', 'LGG-311', 'LGG-314', 'LGG-321', 'LGG-346', 'LGG-519', 'LGG-234', 'LGG-293', 'LGG-532', 'LGG-552', 'LGG-591', 'LGG-594', 'LGG-609', 'LGG-610', 'LGG-613', 'LGG-624', 'LGG-647', 'LGG-766', 'LGG-371']

def classification_list():
    trf = pd.read_csv(train_file)
    pymrmr.mRMR(trf, 'MID', 5)
    train_features = trf.values.tolist()
    tef = pd.read_csv(test_file)
    test_features = tef.values.tolist()
    #cf = pd.read_csv(feature_file_2)
    pymrmr.mRMR(tef, 'MID', 5)
    #og_features_T2 = cf.values.tolist()
    #og_features_T1 = og_features_T1 + og_features_T2
    #bf = pd.read_csv(feature_file_3)
    #og_features_T3 = bf.values.tolist()
    #df = pd.read_csv(feature_file_4)
    #og_features_T4 = bf.values.tolist()
    #ef = pd.read_csv(feature_file_5)
    #og_features_T5 = bf.values.tolist()
    #ff = pd.read_csv(feature_file_6)
    #og_features_T6 = bf.values.tolist()
    #og_features = pd.read_csv(feature_file)
    #og_features = og_features.values.tolist()
    #for list in train_features:
    #    list.pop(0)
    #    list.pop(1)
    #    list.pop(1)
    #for list in test_features:
    #    list.pop(0)
    #    list.pop(1)
    #    list.pop(1)
    #for list in og_features_T3:
    #    list.pop(0)
    #    list.pop(1)
    #    list.pop(1)
    #for list in og_features_T4:
    #    list.pop(0)
    #    list.pop(1)
    #    list.pop(1)
    #for list in og_features_T5:
    #    list.pop(0)
    #    list.pop(1)
    #    list.pop(1)
    #for list in og_features_T6:
    #    list.pop(0)
    #    list.pop(1)
    #    list.pop(1)
    #og_features = []
    #for f in range(len(og_features_T1)):
    #    og_features.append(og_features_T1[f] + og_features_T2[f])
    #og_features_1 = []
    #for f in range(len(og_features_T3)):
    #    og_features_1.append(og_features_T3[f] + og_features_T4[f])
    #og_features_2 = []
    #for j in range(len(og_features_T5)):
    #    og_features_2.append(og_features_T5[j] + og_features_T6[j])
    #og_features_2 = []
    #for i in range(len(og_features_T2)):
    #    og_features_2.append(og_features_T2[i] + og_features_T4[i] + og_features_T6[i])
    #og_features_1 = []
    #for i in range(len(og_features_T1)):
    #    og_features_1.append(og_features_T1[i] + og_features_T3[i] + og_features_T5[i])
    #og_features = []
    #for i in range(len(og_features_1)):
    #    og_features.append(og_features_1[i] + og_features_2[i])
    classification_lister_train = []
    for img in os.listdir(dataPath):
        image_file = os.fsencode(img)
        ImageID = (image_file.decode('utf-8')).split('_')[0]
        if(ImageID in deletion_nums):
            classification_lister_train.append(1)
        elif(('correlations' in ImageID) or ('selected' in ImageID) or ('slices' in ImageID) or ('all' in ImageID)):
            pass
        else:
            classification_lister_train.append(0)
    classification_lister_test = []
    for img in os.listdir(testPath):
        image_file = os.fsencode(img)
        ImageID = (image_file.decode('utf-8')).split('_')[0]
        if(ImageID in deletion_nums):
            classification_lister_test.append(1)
        elif(('correlations' in ImageID) or ('selected' in ImageID) or ('slices' in ImageID)):
            pass
        else:
            classification_lister_test.append(0)
    return [[train_features, test_features], [classification_lister_train, classification_lister_test]]
    #return [[og_features_T1, og_features_T2, og_features_T3, og_features_T4, og_features_T5, og_features_T6], classification_lister]
    #return [[og_features, og_features_1, og_features_2], classification_lister]
    #return [[og_features_T1, og_features_T2], classification_lister]

def k_train_test(og_features, classification_lister):
    X = np.array(og_features[0])
    test = np.array(og_features[1])
    #X1 = np.array(og_features[0])
    #X2 = np.array(og_features[1])
    #X3 = np.array(og_features[2])
    #X4 = np.array(og_features[3])
    #X5 = np.array(og_features[4])
    #X6 = np.array(og_features[5])
    y = np.array(classification_lister[0])
    test_y = np.array(classification_lister[1])
    scaler = RobustScaler()
    #transformer = FastICA(n_components=5)
    #kf = KFold(n_splits=10, shuffle=True)
    #train_total = 0
    #test_total = 0
    #C_total = 0
    #for train, test in kf.split(X):
    #    scaler = RobustScaler()
    #    transformer = FastICA(n_components=5)
    #    #X1_train, X1_test, X2_train, X2_test, y_train, y_test = [], [], [], [], [], []
    #    X_train, X_test, y_train, y_test = [], [], [], []
    #    #X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, y_train, y_test = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    #    for i in train:
    #        X_train.append(X[i])
    #        #X2_train.append(X2[i])
    #        #X3_train.append(X3[i])
    #        #X4_train.append(X4[i])
    #        #X5_train.append(X5[i])
    #        #X6_train.append(X6[i])
    #        y_train.append(y[i])
    #    for i in test:
    #        X_test.append(X[i])
    #        #X2_test.append(X2[i])
    #        #X3_test.append(X3[i])
    #        #X4_test.append(X4[i])
    #        #X5_test.append(X5[i])
    #        #X6_test.append(X6[i])
    #        y_test.append(y[i])
    X_train = scaler.fit_transform(np.array(X))
    #X_train = transformer.fit_transform(np.array(X))
    test = scaler.transform(np.array(test))
    #test = transformer.transform(np.array(test))
        #X1_train = scaler.fit_transform(np.array(X1_train))
        #X1_train = transformer.fit_transform(np.array(X1_train))
        #X1_test = scaler.transform(np.array(X1_test))
        #X1_test = transformer.transform(np.array(X1_test))
        #X2_train = scaler.fit_transform(np.array(X2_train))
        #X2_train = transformer.fit_transform(np.array(X2_train))
        #X2_test = scaler.transform(np.array(X2_test))
        #X2_test = transformer.transform(np.array(X2_test))
        #X_train = []
        #for f in range(len(X1_train)):
        #    X_train.append(list(X1_train[f]) + list(X2_train[f]))
        #X_test = []
        #for f in range(len(X1_test)):
        #    X_test.append(list(X1_test[f]) + list(X2_test[f]))
        #X_train = np.array(X_train)
        #X_test = np.array(X_test)
        #X1_train = scaler.fit_transform(X1_train)
        #X1_test = scaler.transform(X1_test)
        #X2_train = scaler.fit_transform(X2_train)
        #X2_test = scaler.transform(X2_test)
        #X1_train = transformer.fit_transform(X1_train)
        #X1_test = transformer.transform(X1_test)
        #X2_train = transformer.fit_transform(X2_train)
        #X2_test = transformer.transform(X2_test)
        #X3_train = scaler.fit_transform(X3_train)
        #X3_test = scaler.transform(X3_test)
        #X3_train = transformer.fit_transform(X3_train)
        #X3_test = transformer.transform(X3_test)
        #X4_train = scaler.fit_transform(X4_train)
        #X4_test = scaler.transform(X4_test)
        #X4_train = transformer.fit_transform(X4_train)
        #X4_test = transformer.transform(X4_test)
        #X5_train = scaler.fit_transform(X5_train)
        #X5_test = scaler.transform(X5_test)
        #X5_train = transformer.fit_transform(X5_train)
        #X5_test = transformer.transform(X5_test)
        #X6_train = scaler.fit_transform(X6_train)
        #X6_test = scaler.transform(X6_test)
        #X6_train = transformer.fit_transform(X6_train)
        #X6_test = transformer.transform(X6_test)
        #X1_tr = []
        #for i in range(len(X1_train)):
        #    X1_tr.append(X1_train[i] + X2_train[i])
        #X1_te = []
        #for i in range(len(X1_test)):
        #    X1_te.append(X1_test[i] + X2_test[i])
        #X3_tr = []
        #for i in range(len(X3_train)):
        #    X3_tr.append(X3_train[i] + X4_train[i])
        #X3_te = []
        #for i in range(len(X3_test)):
        #    X3_te.append(X3_test[i] + X4_test[i])
        #X5_tr = []
        #for i in range(len(X5_train)):
        #    X5_tr.append(X5_train[i] + X6_train[i])
        #X5_te = []
        #for i in range(len(X5_test)):
        #    X5_te.append(X5_test[i] + X6_test[i])
        #X_train = list(X1_tr) + list(X3_tr) + list(X5_tr)
        #X_train = np.array(X_train)
        #X_test = np.array(list(X1_te) + list(X3_te) + list(X5_te))
        #y_train = np.array(y_train + y_train + y_train)
        #y_test = np.array(y_test + y_test + y_test)
    y_train = np.array(y)
    y_test = np.array(test_y)
    parameters = {'gamma': ('scale', 'auto'), 'C':[1,10]}
    classifier = SVC(kernel='poly')
    gridder = GridSearchCV(classifier, parameters, cv=10, refit=True)
        #parameters = {'n_estimators': [1, 10, 25, 50, 100], 
        #              'max_features':['auto', 'sqrt', 'log2']}
        #classifier = RandomForestClassifier()
        #gridder = GridSearchCV(classifier, parameters, cv=10, refit=True)
    gridder.fit(X_train, y_train)
        #print(y_test)
        #print(gridder.best_params_)
    param_set = gridder.best_params_
        #for key in param_set.keys():
        #    print(key)
        #    print(param_set[key])
    classifier.set_params(**param_set)
        #print(classifier.get_params())
    classifier.fit(X_train, y_train)
    results = []
    test_results = classifier.predict(test)
    test_num = 0
    for i in range(len(test)):
        if (test_results[i] == y_test[i]):
            test_num += 1
    #train_results = classifier.predict(X_train)
    #train_num = 0
    #for i in range(len(X_train)):
    #    if(train_results[i] == y_train[i]):
    #        train_num += 1
    #train_total += train_num/(len(y_train))
    test_total = test_num/(len(y_test))
    #C_total += classifier.get_params()['C']
    #return [(test_total/10), (train_total/10)]
    #return [(test_total/10), (train_total/10), (C_total/10)]
    return test_total


def train_test_run(og_features, classification_lister):
    #print("generated classification classes")
    #print(classification_lister)
    X = np.array(og_features[0])
    #X2 = np.array(og_features[1])
    #X3 = np.array(og_features[2])
    #X4 = np.array(og_features[3])
    #X5 = np.array(og_features[4])
    #X6 = np.array(og_features[5])
    #print(X.shape)
    #X = np.array(og_features)
    #print(X.shape)
    y = np.array(classification_lister)
    #print(y.shape)
    scaler = RobustScaler()
    #transformer = FastICA(n_components=5)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train = scaler.fit_transform(X_train)
    #X_train = transformer.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    #X_test = transformer.fit_transform(X_test)
    #X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, X5_test, X6_train, X6_test, y_train, y_test = train_test_split(X1, X2, X3, X4, X5, X6, y, test_size=0.1)
    #X1_train = scaler.fit_transform(X1_train)
    #X1_test = scaler.transform(X1_test)
    #X2_train = scaler.fit_transform(X2_train)
    #X2_test = scaler.transform(X2_test)
    #X1_train = transformer.fit_transform(X1_train)
    #X1_test = transformer.transform(X1_test)
    #X2_train = transformer.fit_transform(X2_train)
    #X2_test = transformer.transform(X2_test)
    #X3_train = scaler.fit_transform(X3_train)
    #X3_test = scaler.transform(X3_test)
    #X3_train = transformer.fit_transform(X3_train)
    #X3_test = transformer.transform(X3_test)
    #X4_train = scaler.fit_transform(X4_train)
    #X4_test = scaler.transform(X4_test)
    #X4_train = transformer.fit_transform(X4_train)
    #X4_test = transformer.transform(X4_test)
    #X5_train = scaler.fit_transform(X5_train)
    #X5_test = scaler.transform(X5_test)
    #X5_train = transformer.fit_transform(X5_train)
    #X5_test = transformer.transform(X5_test)
    #X6_train = scaler.fit_transform(X6_train)
    #X6_test = scaler.transform(X6_test)
    #X6_train = transformer.fit_transform(X6_train)
    #X6_test = transformer.transform(X6_test)
    #Xa_train = []
    #for f in range(len(X1_train)):
    #    Xa_train.append(list(X1_train[f]) + list(X2_train[f]))
    #Xa_test = []
    #for f in range(len(X1_test)):
    #    Xa_test.append(list(X1_test[f]) + list(X2_test[f]))
    #Xb_train = []
    #for f in range(len(X3_train)):
    #    Xb_train.append(list(X3_train[f]) + list(X4_train[f]))
    #Xb_test = []
    #for f in range(len(X3_test)):
    #    Xb_test.append(list(X3_test[f]) + list(X4_test[f]))
    #Xc_train = []
    #for f in range(len(X5_train)):
    #    Xc_train.append(list(X5_train[f]) + list(X6_train[f]))
    #Xc_test = []
    #for f in range(len(X5_test)):
    #    Xc_test.append(list(X5_test[f]) + list(X6_test[f]))
    #X_train = Xa_train + Xb_train + Xc_train
    #X_train = np.array(X_train)
    #X_test = Xa_test + Xb_test + Xc_test
    #X_test = np.array(X_test)
    #y_train = np.array(list(y_train) + list(y_train) + list(y_train))
    #y_test = np.array(list(y_test) + list(y_test) + list(y_test))
    #print(X_train.shape)
    #X_train = X1_train + X2_train
    #print(X_train.shape)
    #X_test = X1_test + X2_test
    #X_test = transformer.transform(X1_test)
    #X_test = X_test + transformer.transform(X2_test)
    #X_test = scaler.transform(X_test)
    parameters = {'gamma': ('scale', 'auto'), 'C':[1,10]}
    classifier = SVC(kernel='linear')
    gridder = GridSearchCV(classifier, parameters, cv=10, refit=True)
    #print(X_train)
    #print(y_train)
    gridder.fit(X_train, y_train)
    #print(y_test)
    #print(gridder.best_params_)
    param_set = gridder.best_params_
    #for key in param_set.keys():
    #    print(key)
    #    print(param_set[key])
    classifier.set_params(**param_set)
    #print(classifier.get_params())
    classifier.fit(X_train, y_train)
    results = []
    #for obj in X_train:
    #    results.append((classifier.predict([obj]))[0])
    #print(results)
    #results_np = np.array(results)
    #oresults_np = np.array(classifier.predict(X_train))
    #print(results_np)
    #print(oresults_np)
    #print(np.equal(results_np, oresults_np))
    test_results = classifier.predict(X_test)
    test_num = 0
    for i in range(len(X_test)):
        if (test_results[i] == y_test[i]):
            test_num += 1
    #print(test_results)
    #print(y_test)
    train_results = classifier.predict(X_train)
    train_num = 0
    for i in range(len(X_train)):
        if(train_results[i] == y_train[i]):
            train_num += 1
    return [(test_num/(len(y_test))), (train_num/(len(y_train))), classifier.get_params()['C']]
    #print(cross_val_score(classifier, X, y, cv=10, scoring='accuracy'))

feature_classes = classification_list()
#print(train_test_run(feature_classes[0], feature_classes[1]))
result = k_train_test(feature_classes[0], feature_classes[1])
#while(result <= 0.7):
#    result = result = k_train_test(feature_classes[0], feature_classes[1])
print(result)
#test_results = []
#train_results = []
#C_param = []
#C_total = 0
#for c in range(0,10):
#    test_total = 0
#    train_total = 0
#    for i in range(0,10):
#        train_test = k_train_test(feature_classes[0], feature_classes[1])
#        test_total += train_test[0]
#        train_total += train_test[1]
#        C_param.append(train_test[2])
#        C_total += train_test[2]
#    test_results.append(train_test[0])
#    train_results.append(train_test[1])
#print(C_param)
#print(C_total/100)
#print(test_results)
#print(train_results)