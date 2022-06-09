import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

SEED = 2021 #random seed #
num_CV = 5  # number of crossvalidation #
n_splits = 5
repeat = 100
test_size = 0.3 # test set ratio #
max_iter = 100000 # max iteration for SVM #
alphas = np.logspace(-4, 4, 17)

X_ADC = pd.read_csv("results/binWidth/ADC_binWidth_32.csv")
X_T1 = pd.read_csv("results/binWidth/T1GD_binWidth_32.csv") 
X_T2 = pd.read_csv("results/binWidth/T2_binWidth_32.csv")

X_full = X_ADC.merge(X_T1, on='ID').merge(X_T2, on='ID').iloc[:, 1:] #full set
y = pd.read_csv("answer/answerADC.csv").iloc[:,1 ]

datasets = ['X_adc_T1_T2', 'X_T1', 'X_T2', 'X_adc_T1', 'X_adc_T2', 'X_adc', 'X_T1_T2']
oversamples = ['original', 'ros', 'smote', 'adasyn', 'bsmote', 'svmsmote'] #'ksmote', 
methods = ['linear', 'logistic', 'ridge', 'lasso', 'elastic', 'decisiontree', 'randomforest', 'adaboost', 'svm-linear', 'svm-poly', 'svm-rbf', 'fullyconnected']
col_names = []
for o in oversamples:
    for m in methods:
        col_names.append("{}-{}".format(o, m))
for d, dataset in enumerate(datasets):
    print ("[ {} / {} ]".format(d, len(datasets)), "with dataset : ", dataset, )
    if dataset == 'X_adc_T1_T2':
        X = X_full
    elif dataset == 'X_T1':
        X = X_full.iloc[:, :107]
        X_test = X_full.iloc[:, :107]
    elif dataset == 'X_T2':
        X = X_full.iloc[:,107:214]
        X_test = X_full.iloc[:,107:214]
    elif dataset == 'X_adc_T1':
        X = X_full.iloc[:, :214]
    elif dataset == 'X_adc_T2':
        X = X_full.iloc[:, np.r_[0:107, 214:321]]
    elif dataset == 'X_adc':
        X = X_full.iloc[:, :107]
    elif dataset == 'X_T1_T2':
        X = X_full.iloc[:,107:321]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y, random_state=SEED)
    df_train_result = pd.DataFrame([], columns = col_names)
    df_test_result = pd.DataFrame([], columns = col_names)
    pbar = tqdm(total=repeat*n_splits*len(oversamples)*len(methods))
    for seed in range(repeat):
        col_train_values = []
        col_test_values = []
        ros = RandomOverSampler(sampling_strategy='auto', random_state=seed)
        sm = SMOTE(sampling_strategy='auto', random_state=seed)
        adas = ADASYN(sampling_strategy='auto', random_state=seed)
        bsm = BorderlineSMOTE(sampling_strategy='auto', random_state=seed)
        ksm = KMeansSMOTE(sampling_strategy='auto', random_state=seed)
        ssm = SVMSMOTE(sampling_strategy='auto', random_state=seed)
        for _, oversample in enumerate(oversamples):
            if oversample == 'original':
                X_sampled, y_sampled = X_train, y_train
            elif oversample == 'ros':
                X_sampled, y_sampled = ros.fit_resample(X_train,y_train)
            elif oversample == 'smote':
                X_sampled, y_sampled = sm.fit_resample(X_train,y_train)
            elif oversample == 'adasyn':
                X_sampled, y_sampled = adas.fit_resample(X_train,y_train)
            elif oversample == 'bsmote':
                X_sampled, y_sampled = bsm.fit_resample(X_train,y_train)
            elif oversample == 'ksmote':
                X_sampled, y_sampled = ksm.fit_resample(X_train,y_train)
            elif oversample == 'svmsmote':
                X_sampled, y_sampled = ssm.fit_resample(X_train,y_train)
            for _, method in enumerate(methods):
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                AUC_trains = 0
                AUC_tests = 0
                for _, (train, val) in enumerate(cv.split(X_sampled, y_sampled)):
                    if method == 'fullyconnected':
                        sc = StandardScaler()
                        X_stand = sc.fit_transform(X_sampled)
                        X_stand_test = sc.transform(X_test)
                        classifier = Sequential()
                        classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
                        classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
                        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
                        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                        classifier.fit(X_stand[train], y_sampled.to_numpy()[train], batch_size = 10, epochs = 2)
                        preds_train = classifier.predict(X_stand)
                        preds_test = classifier.predict(X_stand_test)
                    else:
                        if method == 'linear':
                            classifier = LinearRegression()
                        elif method == 'logistic':
                            classifier = LogisticRegressionCV(cv = num_CV, random_state=seed)
                        elif method == 'ridge':
                            classifier = RidgeCV(alphas = alphas, cv = num_CV)
                        elif method =='lasso':
                            classifier = LassoCV(alphas = alphas, cv = num_CV, random_state=seed)
                        elif method =='elastic':
                            classifier = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv = num_CV, random_state=seed)
                        elif method == 'decisiontree':
                            classifier = tree.DecisionTreeClassifier()
                        elif method == 'randomforest':
                            classifier = RandomForestClassifier(max_depth=4, random_state=seed)
                        elif method == 'adaboost':
                            classifier = AdaBoostClassifier(n_estimators=1000, random_state=seed)
                        elif method == 'svm-linear':
                            classifier = svm.SVC(kernel='linear', max_iter = max_iter)
                        elif method == 'svm-poly':
                            classifier = svm.SVC(kernel='poly', max_iter = max_iter)
                        elif method == 'svm-rbf':
                            classifier = svm.SVC(kernel='rbf', max_iter = max_iter)
                        classifier.fit(X_sampled.to_numpy()[train], y_sampled.to_numpy()[train])
                        preds_train = classifier.predict(X_sampled)
                        preds_test = classifier.predict(X_test)
                    performance_train = roc_auc_score(y_sampled, preds_train)
                    performance_test = roc_auc_score(y_test, preds_test)
                    AUC_trains += performance_train
                    AUC_tests += performance_test
                    pbar.update(1)
                col_train_values.append(AUC_trains/n_splits)
                col_test_values.append(AUC_tests/n_splits)
        df_train_result.loc[seed] = col_train_values
        df_test_result.loc[seed] = col_test_values
    df_train_result.to_excel("{}-{}-train.xls".format(dataset, repeat))
    print("saved: ", "{}-{}-train.xls".format(dataset, repeat))
    df_test_result.to_excel("{}-{}-test.xls".format(dataset, repeat))
    print("saved: ", "{}-{}-test.xls".format(dataset, repeat))