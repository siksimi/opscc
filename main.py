"""
Multiparametric MRI-based radiomics model for predicting human papillomavirus status in oropharyngeal squamous cell carcinoma: optimization using oversampling and machine learning techniques
@author: Yongsik Sim

"""


# import libraries #
"""
sklearn (https://scikit-learn.org/)
imblearn (https://imbalanced-learn.org/)

"""

import numpy as np
import pandas as pd
import warnings, os, datetime
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
warnings.filterwarnings(action='ignore')


##### function definition for feature selection #####

# K-best selection filter using ANOVA-F statistics (K-best)
def kbest(X, y, X_test, k):   
    selector = SelectKBest(f_regression, k=k)
    X = selector.fit_transform(X, y)
    X_test = selector.transform(X_test)
    return X, X_test
    
# high correlation filter 
def corr_filter(X, X_test, threshold):
    corr_matrix = X.corr() # Create correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))  # Select upper triangle of correlation matrix
    # Find index of feature columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)] 
    X = X.drop(columns = to_drop, axis=1)
    X_test = X_test.drop(columns = to_drop, axis=1)

    return X, X_test
# principal component analysis (PCA)
def pca(X, y, X_test, n_components): 
    selector = PCA(n_components = n_components)
    X = selector.fit_transform(X, y)
    X_test = selector.transform(X_test)      
    return X, X_test

##### Global variables #####

SEED = 2022                         # random seed #
num_CV = 5                          # number of cross-validation to select optimal parameters (Ridge, LASSO, and Elastic Net) #
n_splits = 5                        # number of cross-validation was applied to validate model performance in the training set # 
repeat = 100                        # total training iterations for each combination of oversampling and machine learning techniques #
max_iter = 100000                   # max iteration for SVM #
alphas = np.logspace(-4, 4, 17)     # range of alpha for Ridge, LASSO, and Elastic Net #


##### Combination of datasets, feature selection, oversampling, and machine learning techniques #####

datasets = ['X_adc_T1_T2', 'X_T1', 'X_T2', 'X_adc_T1', 'X_adc_T2', 'X_adc', 'X_T1_T2']
feature_selections = ['corr_0.4', 'corr_0.6', 'corr_0.8', 'kbest_10', 'kbest_20', 'kbest_30', 'pca_5', 'pca_10', 'pca_15', 'full'] 
oversamples = ['original', 'ros', 'smote', 'adasyn', 'bsmote', 'svmsmote'] 
methods = ['linear', 'logistic', 'ridge', 'lasso', 'elastic', 'decisiontree', 'randomforest', 'adaboost', 'svm-linear', 'svm-poly', 'svm-rbf', 'fullyconnected']


### column names for the output file 
col_names = []
for o in oversamples:
    for m in methods:
        col_names.append("{}-{}".format(o, m))

### output folder
output_dir = os.path.join(os.getcwd(), f"output/{datetime.datetime.now().strftime('-%H-%M-%S')}")
os.makedirs(output_dir)


##### Load data and define training and validation sets #####

df = pd.read_csv("data/merged.csv") # Reading a file containing entire radiomics feature values

training_df = df.loc[df['Outside'] == 0]    # Training dataset
test_df = df.loc[df['Outside'] == 1]        # External validation dataset

X_train_full = training_df.iloc[:, 5:]      # Features for training
y_train = training_df.iloc[:, 1 ]           # labels 
X_test_full = test_df.iloc[:, 5:]           # Features for validation
y_test = test_df.iloc[:, 1 ]                # labels


##### Main #####

for d, dataset in enumerate(datasets):                       ## Use one of the following datasets.
    for f, feature_selection in enumerate(feature_selections):           
        if dataset == 'X_T1':                                       # CE-T1WI features
            X_train = X_train_full.iloc[:, :107]
            X_test = X_test_full.iloc[:, :107]
        elif dataset == 'X_T2':                                     # T2WI features
            X_train = X_train_full.iloc[:,107:214]
            X_test = X_test_full.iloc[:,107:214]
        elif dataset == 'X_adc_T1_T2':                              # ADC + CE-T1WI + T2WI features
            X_train = X_train_full
            X_test = X_test_full
        elif dataset == 'X_adc_T1':                                 # ADC + CE-T1WI features
            X_train = X_train_full.iloc[:, np.r_[0:107, 214:321]]
            X_test = X_test_full.iloc[:, np.r_[0:107, 214:321]]
        elif dataset == 'X_adc_T2':                                 # ADC + T2WI features
            X_train = X_train_full.iloc[:,107:321]
            X_test = X_test_full.iloc[:,107:321]
        elif dataset == 'X_adc':                                    # ADC features
            X_train = X_train_full.iloc[:, 214:]
            X_test = X_test_full.iloc[:, 214:]
        elif dataset == 'X_T1_T2':                                  # CE-T1WI + T2WI features
            X_train = X_train_full.iloc[:, :214]
            X_test = X_test_full.iloc[:, :214]
                                                            ## Use one of the following feature selection filters.
        if feature_selection == 'full':                             # None (Use of full features)
            X, X_test = X_train, X_test
        elif feature_selection == 'corr_0.4':                       # high correlation filter with the highest allowable correlation of 0.4
            X, X_test = corr_filter(X_train, X_test, 0.4)
        elif feature_selection == 'corr_0.6':                       # high correlation filter with the highest allowable correlation of 0.6
            X, X_test = corr_filter(X_train, X_test, 0.6)
        elif feature_selection == 'corr_0.8':                       # high correlation filter with the highest allowable correlation of 0.8
            X, X_test = corr_filter(X_train, X_test, 0.8)
        elif feature_selection == 'kbest_10':                       # K-best selection filter using ANOVA-F statistics (K = 10)
            X, X_test = kbest(X_train, y_train, X_test, 10)
        elif feature_selection == 'kbest_20':                       # K-best selection filter using ANOVA-F statistics (K = 20)
            X, X_test = kbest(X_train, y_train, X_test, 20)                        
        elif feature_selection == 'kbest_30':                       # K-best selection filter using ANOVA-F statistics (K = 30)
            X, X_test = kbest(X_train, y_train, X_test, 30)                        
        elif feature_selection == 'pca_5':                          # principal component analysis (PCA) (number of dimensions = 5)
            X, X_test = pca (X_train, y_train, X_test, 5)
        elif feature_selection == 'pca_10':                         # principal component analysis (PCA) (number of dimensions = 10)
            X, X_test = pca (X_train, y_train, X_test, 10)
        elif feature_selection == 'pca_15':                         # principal component analysis (PCA) (number of dimensions = 15)
            X, X_test = pca (X_train, y_train, X_test, 15)        
            
        print ("[ {} / {} ]".format(d, len(datasets)), "with dataset : {} / feature selection {}".format(dataset,feature_selection))                                          
        
        df_train_result = pd.DataFrame([], columns = col_names)
        df_test_result = pd.DataFrame([], columns = col_names)
        pbar = tqdm(total=repeat*n_splits*len(oversamples)*len(methods))
        for seed in range(repeat):                                  ### The entire process will be repeated *repeat* times
            col_train_values = []
            col_test_values = []
            ros = RandomOverSampler(sampling_strategy='auto', random_state=seed)        # Oversamping: Random oversampler
            sm = SMOTE(sampling_strategy='auto', random_state=seed)                     # Oversamping: SMOTE
            adas = ADASYN(sampling_strategy='auto', random_state=seed)                  # Oversamping: ADASYN
            bsm = BorderlineSMOTE(sampling_strategy='auto', random_state=seed)          # Oversamping: Borderline SMOTE
            ksm = KMeansSMOTE(sampling_strategy='auto', random_state=seed)              # Oversamping: K-Means SMOTE
            ssm = SVMSMOTE(sampling_strategy='auto', random_state=seed)                 # Oversamping: SVM SMOTE
            for _, oversample in enumerate(oversamples):                        ## Use one of the following oversampling methods.
                if oversample == 'original':                                            # Oversamping: None
                    X_sampled, y_sampled = X_train, y_train
                elif oversample == 'ros':                                               # Oversamping: Random oversampler
                    X_sampled, y_sampled = ros.fit_resample(X_train, y_train)
                elif oversample == 'smote':                                             # Oversamping: SMOTE
                    X_sampled, y_sampled = sm.fit_resample(X_train, y_train)
                elif oversample == 'adasyn':                                            # Oversamping: ADASYN
                    X_sampled, y_sampled = adas.fit_resample(X_train, y_train)
                elif oversample == 'bsmote':                                            # Oversamping: Borderline SMOTE
                    X_sampled, y_sampled = bsm.fit_resample(X_train, y_train)
                elif oversample == 'ksmote':                                            # Oversamping: K-Means SMOTE
                    X_sampled, y_sampled = ksm.fit_resample(X_train, y_train)
                elif oversample == 'svmsmote':                                          # Oversamping: SVM SMOTE
                    X_sampled, y_sampled = ssm.fit_resample(X_train, y_train)
                for _, method in enumerate(methods):                            ## Use one of the following machine learning methods.
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    AUC_trains = 0
                    AUC_tests = 0
                    for _, (train, val) in enumerate(cv.split(X_sampled, y_sampled)):
                        if method == 'fullyconnected':                                  # Machine learning : Fully connected layer
                            sc = StandardScaler()
                            X_stand = sc.fit_transform(X_sampled)
                            X_stand_test = sc.transform(X_test)
                            classifier = Sequential()
                            classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
                            classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
                            classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
                            classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                            classifier.fit(X_stand[train], y_sampled.to_numpy()[train], batch_size = 10, epochs = 2, verbose = 0)
                            preds_train = classifier.predict(X_stand)
                            preds_test = classifier.predict(X_stand_test)
                        else:
                            if method == 'linear':                                      # Machine learning : linear regression
                                classifier = LinearRegression()
                            elif method == 'logistic':                                  # Machine learning : logistic regression
                                classifier = LogisticRegressionCV(cv = num_CV, random_state=seed)
                            elif method == 'ridge':                                     # Machine learning : ridge 
                                classifier = RidgeCV(alphas = alphas, cv = num_CV)
                            elif method =='lasso':                                      # Machine learning : LASSO
                                classifier = LassoCV(alphas = alphas, cv = num_CV, random_state=seed)
                            elif method =='elastic':                                    # Machine learning : Elastic Net with L1:L2 = 1:1
                                classifier = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv = num_CV, random_state=seed)
                            elif method == 'decisiontree':                              # Machine learning : Decision tree classifier
                                classifier = tree.DecisionTreeClassifier()
                            elif method == 'randomforest':                              # Machine learning : Random Forest
                                classifier = RandomForestClassifier(max_depth=4, random_state=seed)
                            elif method == 'adaboost':                                  # Machine learning : Adaboost
                                classifier = AdaBoostClassifier(n_estimators=1000, random_state=seed)
                            elif method == 'svm-linear':                                # Machine learning : SVM with linear kernel
                                classifier = svm.SVC(kernel='linear', max_iter = max_iter)
                            elif method == 'svm-poly':                                  # Machine learning : SVM with polynormial kernel
                                classifier = svm.SVC(kernel='poly', max_iter = max_iter)
                            elif method == 'svm-rbf':                                   # Machine learning : SVM with rbf kernel
                                classifier = svm.SVC(kernel='rbf', max_iter = max_iter)
                            classifier.fit(X_sampled.to_numpy()[train], y_sampled.to_numpy()[train])
                            preds_train = classifier.predict(X_sampled)
                            preds_test = classifier.predict(X_test)
                        performance_train = roc_auc_score(y_sampled, preds_train)       # Calculation of training AUC
                        performance_test = roc_auc_score(y_test, preds_test)            # Calculation of validation AUC
                        AUC_trains += performance_train
                        AUC_tests += performance_test
                        pbar.update(1)
                    col_train_values.append(AUC_trains/n_splits)
                    col_test_values.append(AUC_tests/n_splits)
            df_train_result.loc[seed] = col_train_values
            df_test_result.loc[seed] = col_test_values
        df_train_result.to_excel(os.path.join(output_dir, "{}-{}-{}-train.xlsx".format(dataset, feature_selection, repeat)))     # Saving AUC values for the training set
        print("saved: ", "{}-{}-{}-train.xlsx".format(dataset, feature_selection, repeat))
        df_test_result.to_excel(os.path.join(output_dir, "{}-{}-{}-test.xlsx".format(dataset, feature_selection, repeat)))       # Saving AUC values for the validation set
        print("saved: ", "{}-{}-{}-test.xlsx".format(dataset, feature_selection, repeat))        