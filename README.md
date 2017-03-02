# Identify-Fraud-from-Enron-Email
In this project I played detective and put my machine learning skills to use by building an algorithm to identify Enron Employees who may
have committed fraud based on the public Enron financial and email dataset. Starter code was provided by Udacity.



* Final Project Write-up: Q_A.ipynb
* Pickle files: my_dataset.pkl, my_classifier.pkl, my_feature_list.pkl
* Machine Learning file: poi_id.py (run this file if needed)
* Tester file: tester.py (unmodified from Udacity-distributed code)
* feautere formatting: feature_format.py
* References: Reference.txt 

I have also attached my workflow as documentation.html .It can be downloaded to view .

Final result :

Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                          ('selection', SelectKBest(k=10, score_func=<function f_classif at 0x00000000071B2278>)),
                                          ('pca', PCA(copy=True, n_components=7, whiten=False)), ('naive_bayes', GaussianNB())])
