# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
import numpy as np
import pprint
from collections import defaultdict
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
'''
The below are the features our dataset contains .But we all not be using all the features provided for further analysis.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                    'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                    'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                    'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
                'from_this_person_to_poi', 'shared_receipt_with_poi']
                
POI label: [‘poi’] (boolean, represented as integer)
'''



initial_features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                    'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                    'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                    'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                    'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
number_of_poi = 0
number_of_non_poi = 0
null_count = defaultdict(int)
for name, features in data_dict.iteritems():
    if features['poi']:
        number_of_poi += 1
    else:
        number_of_non_poi += 1
    for key,value in features.iteritems():
        if value == 'NaN':
            null_count[key] += 1
            
            
    

#AS we know the name of chairman of Enron Corporation we can use his name 
number_of_features = len(data_dict['LAY KENNETH L'])

print("Total number of persons in the data set: {}".format(len(data_dict)))
print("Total number of persons of interest (poi) in the data set : {}".format(number_of_poi))
print("Total number of non persons of interest (non poi) in the data set : {}".format(number_of_non_poi))
print("Each person has {} features and are listed below :\n".format(number_of_features))
for key in data_dict['LAY KENNETH L'].keys():
    print key
print "\nTotal number of missing values in features:"
for key in null_count.keys():
    print(key,null_count[key])



###  Remove outliers


'''
Now I will take a look at my data for checking outliers .And I will select two features i.e 'salary' and 'bonus'.
I will plot them so that we ca find if there are any outliers.
'''
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
### plot features
import matplotlib.pyplot as plt
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
print "\nChecking for outliers"
plt.show()


#Checking mannually I found 'TOTAL' and I will remove it ,then I will proceed further to check for more outliers.
data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)#it was no way related to our motto so I removed
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
print "\n After removal of outlier 'TOTAL' the scenario is : "
plt.show()

outliers = []
for key in data_dict:
    value = data_dict[key]['salary']
    if value == 'NaN':
        continue
    outliers.append((key, int(value)))

Top_outliers = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
print "The top 4 outliers are :\n"
print Top_outliers
#These outliers belong to our persion of interest .Thus I will not remove these .





### Store for easy export below.
initial_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(initial_dataset, initial_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#Scaling features
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
scaled_features = min_max_scaler.fit_transform(features)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(scaled_features, labels, test_size=0.1, random_state=42)


#Trying a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from time import time


#Defining a function that will receive a type of classifier and will predict after classifying
def classify(clf):
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Accuracy :",accuracy_score(pred, labels_test)
    print "Precission Score :",precision_score(pred, labels_test)
    print "Recall Score :",recall_score(pred, labels_test)

print "\n==============Before addition of new features==============="
print "\nChecking for GaussianNB :"
classify(GaussianNB())

print "\nChecking for DecisionTreeClassifier"
classify(DecisionTreeClassifier())

print "\nChecking for SVC"
classify(SVC())

print "\nChecking for RandomForestClassifier"
classify(RandomForestClassifier())


###  Create new feature(s)

# I am creating two new features.
for name, fetaures in data_dict.items():
    if fetaures['from_messages'] == 'NaN' or fetaures['from_this_person_to_poi'] == 'NaN':
        fetaures['fraction_from_this_person_to_poi'] = 0.0
    else:
        fetaures['fraction_from_this_person_to_poi'] = \
                                    fetaures['from_this_person_to_poi'] / float(fetaures['from_messages'])

    if fetaures['to_messages'] == 'NaN' or fetaures['from_poi_to_this_person'] == 'NaN':
        fetaures['fraction_from_poi_to_this_person'] = 0.0
    else:
        fetaures['fraction_from_poi_to_this_person'] = \
                                    fetaures['from_poi_to_this_person'] / float(fetaures['to_messages'])
            


# Now I will follow the same procedure as before to calculate performance after new features.
my_dataset = data_dict

#Adding my new features into the initial feature list
initial_features_list.extend(['fraction_from_poi_to_this_person','fraction_from_this_person_to_poi'])


data_ = featureFormat(my_dataset, initial_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data_)


#Scaling features
min_max_scaler = preprocessing.MinMaxScaler()
scaled_features = min_max_scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Now lets see how the perfomance gets affected after addition of new features

print "\n==============After addition of new features==============="
print "\nChecking for GaussianNB :"
classify(GaussianNB())

print "\nChecking for DecisionTreeClassifier"
classify(DecisionTreeClassifier())

print "\nChecking for SVC"
classify(SVC())

print "\nChecking for RandomForestClassifier"
classify(RandomForestClassifier())


# score function
def score_func(y_true,y_predict):
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for prediction, truth in zip(y_predict, y_true):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        else:
            true_positives += 1
    if true_positives == 0:
        return (0,0,0)
    else:
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        return (precision,recall,f1)

# univariateFeatureSelection function
def univariateFeatureSelection(f_list, my_dataset):
    result = []
    for feature in f_list:
        # Replace 'NaN' with 0
        for name in my_dataset:
            data_point = my_dataset[name]
            if not data_point[feature]:
                data_point[feature] = 0
            elif data_point[feature] == 'NaN':
                data_point[feature] =0

        data = featureFormat(my_dataset, ['poi',feature], sort_keys = True, remove_all_zeroes = False)
        labels, features = targetFeatureSplit(data)
        features = [abs(x) for x in features]
        from sklearn.cross_validation import StratifiedShuffleSplit
        cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for train_idx, test_idx in cv:
            for ii in train_idx:
                features_train.append( features[ii] )
                labels_train.append( labels[ii] )
            for jj in test_idx:
                features_test.append( features[jj] )
                labels_test.append( labels[jj] )
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        score = score_func(labels_test,predictions)
        result.append((feature,score[0],score[1],score[2]))
    result = sorted(result, reverse=True, key=lambda x: x[3])
    return result 

#Univariate feature selection
univariate_result = univariateFeatureSelection(initial_features_list,my_dataset)
print '\n### univariate feature selection result'
for l in univariate_result:
    print l

# My selected feature list
features_list = ['poi','total_stock_value','exercised_stock_options','bonus','deferred_income','long_term_incentive',
                    'restricted_stock','salary','total_payments','other', 'shared_receipt_with_poi']
                         
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

min_max_scaler = preprocessing.MinMaxScaler()
scaled_features = min_max_scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = train_test_split(scaled_features, labels, test_size=0.1, random_state=42)


print "\nChecking for GaussianNB :"
classify(GaussianNB())

print "\nChecking for DecisionTreeClassifier"
classify(DecisionTreeClassifier())

print "\nChecking for SVC"
classify(SVC())

print "\nChecking for RandomForestClassifier"
classify(RandomForestClassifier())



###  Tune your classifier to achieve better than .3 precision and recall 
features_train, features_test, labels_train, labels_test = \
                                    train_test_split(features, labels, test_size=0.3, random_state=42)
    
print "\n==============DecisionTreeClassifier==============="
decision_tree = DecisionTreeClassifier()

dt_params = [{'min_samples_split': [2,3,4], 'criterion': ['gini', 'entropy']}]

dt_grid = GridSearchCV(estimator = decision_tree,\
                       param_grid = dt_params,\
                       cv = StratifiedKFold(labels_train, n_folds = 6, shuffle = True),\
                       n_jobs = -1,\
                       scoring = 'f1')

start_fitting = time()
dt_grid.fit(features_train, labels_train)
end_fitting = time()
print("Training time : {}".format(round(end_fitting - start_fitting, 3)))

start_predicting = time()
dt_pred = dt_grid.predict(features_test)
end_predicting = time()
print("Predicting time : {}".format(round(end_predicting - start_predicting, 3)))

dt_accuracy = accuracy_score(dt_pred, labels_test)
print('Decision Tree accuracy : {}'.format(dt_accuracy))
print "f1 score :",f1_score(dt_pred, labels_test)
print "precision score :",precision_score(dt_pred, labels_test)
print "recall score :",recall_score(dt_pred, labels_test)
print(dt_grid.best_estimator_)



print"\n==============GaussianNB============="

gauss = GaussianNB()
nb_pipe = Pipeline([('scaler', MinMaxScaler()),('selection', SelectKBest()),('pca', PCA()),('naive_bayes', gauss)])

nb_parameters = [{ 'selection__k': [8,9,10], 'pca__n_components': [6,7,8] }]

nb_grid = GridSearchCV(estimator = nb_pipe,\
                        param_grid = nb_parameters,\
                        n_jobs = -1,\
                        cv = StratifiedKFold(labels_train, n_folds = 6, shuffle = True),\
                        scoring = 'f1')

start_fitting = time()
nb_grid.fit(features_train, labels_train)
end_fitting = time()
print("Training time : {}".format(end_fitting - start_fitting))

start_predicting = time()
nb_pred = nb_grid.predict(features_test)
end_predicting = time()
print("Predicting time : {}".format(end_predicting - start_predicting))

nb_accuracy = accuracy_score(nb_pred, labels_test)
print('Naive Bayes accuracy : {}'.format(nb_accuracy))
print "f1 score :",f1_score(nb_pred, labels_test)
print "precision score :",precision_score(nb_pred, labels_test)
print "recall score :",recall_score(nb_pred, labels_test)
print(nb_grid.best_estimator_)


print "\n=================SVC================="

svc_pipe = Pipeline([('scaler',MinMaxScaler()), ('svc', SVC())])

svc_params = { 'svc__kernel': ['linear','rbf'],
                   'svc__C': [0.1,1,10,100,1000],
                   'svc__gamma': [1e-3,1e-4,1e-1,1,10] }
                 

svc_grid = GridSearchCV(estimator = svc_pipe,\
                        param_grid = svc_params,\
                        cv = StratifiedKFold(labels_train, n_folds = 6, shuffle = True),\
                        n_jobs = -1,\
                        scoring = 'f1')

start_fitting = time()
svc_grid.fit(features_train, labels_train)
end_fitting = time()
print("Training time : {}".format(end_fitting - start_fitting))

start_predicting = time()
svc_pred = svc_grid.predict(features_test)
end_predicting = time()
print("Predicting time : {}".format(end_predicting - start_predicting))

svc_accuracy = accuracy_score(svc_pred, labels_test)
print('SVC accuracy score : {}'.format(svc_accuracy))
print "f1 score :",f1_score(svc_pred, labels_test)
print "precision score :",precision_score(svc_pred, labels_test)
print "recall score :",recall_score(svc_pred, labels_test)
svc_best_estimator = svc_grid.best_estimator_
print(svc_best_estimator)


test_classifier(nb_grid.best_estimator_, my_dataset, features_list)

#Checking the affect of new feature on the final classifier

test_features_list = ['poi','total_stock_value','exercised_stock_options','bonus','deferred_income','long_term_incentive',
    'restricted_stock','salary','total_payments','other', 'shared_receipt_with_poi','fraction_from_this_person_to_poi']

print "\n=================Effect of new feature on final classifier================="

test_classifier(nb_grid.best_estimator_, my_dataset, test_features_list)

###Task 6: Dump your classifier, dataset, and features_list so anyone can

dump_classifier_and_data(nb_grid.best_estimator_, my_dataset, features_list)