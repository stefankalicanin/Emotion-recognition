#Loading libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from scipy.stats import yeojohnson
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Loading the database
data = pd.read_csv('emotion_data_gees.csv')


#Data overview
print("Data:\n", data.head(10))


#Description of the data
print("Data format:", data.shape)
print("Number of samples:", data.shape[0])
print("Feature numbers:", data.shape[1])
print("Feature types:\n", data.dtypes)
print("Percentage of missing data:\n", data.isna().sum()/data.shape[0]*100)
print("Initial data description:\n", data.describe())


#Number of samples for each gender
print("Number of samples for male:", data[data["spk"].isin(["MM", "MV", "SK"])].shape[0])
print("Number of samples for female:", data[data["spk"].isin(["OK", "BM", "SZ"])].shape[0])


#Number of samples for each class by gender
print("Number of male in class L:", data[(data['emotion']=="L") & (data['spk'].isin(["MM", "MV", "SK"]))].shape[0])
print("Number of male in class N:", data[(data['emotion']=="N") & (data['spk'].isin(["MM", "MV", "SK"]))].shape[0])
print("Number of male in class R:", data[(data['emotion']=="R") & (data['spk'].isin(["MM", "MV", "SK"]))].shape[0])
print("Number of male in class S:", data[(data['emotion']=="S") & (data['spk'].isin(["MM", "MV", "SK"]))].shape[0])
print("Number of male in class T:", data[(data['emotion']=="T") & (data['spk'].isin(["MM", "MV", "SK"]))].shape[0])

print("Number of female in class L:", data[(data['emotion']=="L") & (data['spk'].isin(["OK", "BM", "SZ"]))].shape[0])
print("Number of female in class N:", data[(data['emotion']=="N") & (data['spk'].isin(["OK", "BM", "SZ"]))].shape[0])
print("Number of female in class R:", data[(data['emotion']=="R") & (data['spk'].isin(["OK", "BM", "SZ"]))].shape[0])
print("Number of female in class S:", data[(data['emotion']=="S") & (data['spk'].isin(["OK", "BM", "SZ"]))].shape[0])
print("Number of female in class T:", data[(data['emotion']=="T") & (data['spk'].isin(["OK", "BM", "SZ"]))].shape[0])


#Removing unnecessary features
data.drop(['name', 'spk'], axis = 1, inplace = True)


#Converting class labels into numerical features
data['emotion'] = data['emotion'].replace({'L':0, 'N':1, 'R':2, 'S':3, 'T':4})


#Number of samples for each class
print("Number of samples in class L:", data[data['emotion']==0].shape[0])
print("Number of samples in class N:", data[data['emotion']==1].shape[0])
print("Number of samples in class R:", data[data['emotion']==2].shape[0])
print("Number of samples in class S:", data[data['emotion']==3].shape[0])
print("Number of samples in class T:", data[data['emotion']==4].shape[0])


#Separate the features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


#Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13, stratify=y)


#Standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train1_std = pd.DataFrame(scaler.transform(X_train))
X_test1_std = pd.DataFrame(scaler.transform(X_test))


#Yeojohnson transformation
transformed_data_training = np.empty_like(X_train1_std)
for i in range(X_train1_std.shape[1]):
    transformed_data_training[:, i], _ = yeojohnson(X_train1_std.iloc[:, i])

transformed_data_test = np.empty_like(X_test1_std)
for i in range(X_test1_std.shape[1]):
    transformed_data_test[:, i], _ = yeojohnson(X_test1_std.iloc[:, i])

X_train_std = pd.DataFrame(transformed_data_training)
X_test_std = pd.DataFrame(transformed_data_test)


#LDA
#Search for best parameters
model = LinearDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
parameters = {'n_components': range(1, 4)}
clf = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=kfold, refit=True, verbose=3)
clf.fit(X_train_std, y_train)

print("Searcing for best parameters... -LDA-")
print(clf.best_score_)
print(clf.best_params_)


#Model
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train_std, y_train)
X_train_lda = lda.transform(X_train_std)
X_test_lda = lda.transform(X_test_std)


#Visualization
X_lda_class0 = X_train_lda[y_train == 0]
X_lda_class1 = X_train_lda[y_train == 1]
X_lda_class2 = X_train_lda[y_train == 2]
X_lda_class3 = X_train_lda[y_train == 3]
X_lda_class4 = X_train_lda[y_train == 4]

plt.scatter(X_lda_class0[:, 0], X_lda_class0[:, 1], color='red', label='Anger')
plt.scatter(X_lda_class1[:, 0], X_lda_class1[:, 1], color='green', label='Neutral')
plt.scatter(X_lda_class2[:, 0], X_lda_class2[:, 1], color='blue', label='Joy')
plt.scatter(X_lda_class3[:, 0], X_lda_class3[:, 1], color='yellow', label='Fear')
plt.scatter(X_lda_class4[:, 0], X_lda_class4[:, 1], color='purple', label='Sadness')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.legend()
plt.show()


#PCA
pca = PCA(n_components=0.95)
pca.fit(X_train_std, y_train)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


#Visualization
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.show()


#Logistic Regression
#Search for best parameters
model = LogisticRegression()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
parameters = {
    'fit_intercept' : [True, False],
    'class_weight' : [None, 'balanced'],
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'multi_class' : ['auto'],
    'max_iter' : [100, 200, 300]
}
clf = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=kfold, refit=True, verbose=3)
clf.fit(X_train_std, y_train)

print("Searcing for best parameters... -LR-")
print(clf.best_score_)
print(clf.best_params_)


#Basic
print("----------------LR-----------------------")
print("\n\n\n")
model_basic_LR = LogisticRegression(C=1,class_weight=None, fit_intercept=False, max_iter=1000, multi_class='auto', solver='saga')
model_basic_LR.fit(X_train_std, y_train)

train_pred = model_basic_LR.predict(X_train_std)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(Logistic Regression - basic) of training set: {accuracy}")
print(f"Precision(Logistic Regression - basic) of training set: {precision}")
print(f"F1-score (Logistic Regression - basic) of training set: {f1}")

test_pred = model_basic_LR.predict(X_test_std)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(Logistic Regression - basic) of test set: {accuracy}")
print(f"Precision(Logistic Regression - basic) of test set: {precision}")
print(f"F1-score (Logistic Regression - basic) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))


#With LDA
model_LDA_LR =  LogisticRegression(C=1,class_weight=None, fit_intercept=False, max_iter=1000, multi_class='auto', solver='saga')
model_LDA_LR.fit(X_train_lda, y_train)

train_pred = model_LDA_LR.predict(X_train_lda)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(Logistic Regression - LDA) of training set: {accuracy}")
print(f"Precision(Logistic Regression - LDA) of training set: {precision}")
print(f"F1-score (Logistic Regression - LDA) of training set: {f1}")

test_pred = model_LDA_LR.predict(X_test_lda)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(Logistic Regression - LDA) of test set: {accuracy}")
print(f"Precision(Logistic Regression - LDA) of test set: {precision}")
print(f"F1-score (Logistic Regression - LDA) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))
print("----------------LR-----------------------")
print("\n\n\n")


#With PCA
model_PCA_LR =  LogisticRegression(C=1,class_weight=None, fit_intercept=False, max_iter=1000, multi_class='auto', solver='saga')
model_PCA_LR.fit(X_train_pca, y_train)

train_pred = model_PCA_LR.predict(X_train_pca)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(Logistic Regression - PCA) of training set: {accuracy}")
print(f"Precision(Logistic Regression - PCA) of training set: {precision}")
print(f"F1-score (Logistic Regression - PCA) of training set: {f1}")

test_pred = model_PCA_LR.predict(X_test_pca)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(Logistic Regression - PCA) of test set: {accuracy}")
print(f"Precision(Logistic Regression - PCA) of test set: {precision}")
print(f"F1-score (Logistic Regression - PCA) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))
print("----------------LR-----------------------")
print("\n\n\n")


#SVM
#Searcing for best parameters
model = SVC()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
parameters = {
    'C' : [0.1,0.25, 1, 1.5],
    'kernel' : ['poly', 'rbf', 'sigmoid'],
    'decision_function_shape' : ['ovo', 'ovr']
}
clf = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=kfold, refit=True, verbose=3)
clf.fit(X_train_std, y_train)

print("Searcing for best parameters... -SVM-")
print(clf.best_score_)
print(clf.best_params_)


#Basic
print("----------------SVM-----------------------")
print("\n\n\n")
model_SVM_basic = SVC(C=0.25, decision_function_shape='ovo', kernel='rbf')
model_SVM_basic.fit(X_train_std, y_train)

train_pred = model_SVM_basic.predict(X_train_std)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(SVM - basic) of training set: {accuracy}")
print(f"Precision(SVM - basic) of training set: {precision}")
print(f"F1-score (SVM - basic) of training set: {f1}")

test_pred = model_SVM_basic.predict(X_test_std)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(SVM - basic) of test set: {accuracy}")
print(f"Precision(SVM - basic) of test set: {precision}")
print(f"F1-score (SVM - basic) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))


#With LDA
model_SVM_LDA = SVC(C=0.25, decision_function_shape='ovo', kernel='rbf')
model_SVM_LDA.fit(X_train_lda, y_train)

train_pred = model_SVM_LDA.predict(X_train_lda)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(SVM - lda) of training set: {accuracy}")
print(f"Precision(SVM - lda) of training set: {precision}")
print(f"F1-score (SVM - lda) of training set: {f1}")

test_pred = model_SVM_LDA.predict(X_test_lda)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(SVM - lda) of test set: {accuracy}")
print(f"Precision(SVM - lda) of test set: {precision}")
print(f"F1-score (SVM - lda) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))
print("----------------SVM-----------------------")
print("\n\n\n")


#With PCA
model_SVM_PCA = SVC(C=0.25, decision_function_shape='ovo', kernel='rbf')
model_SVM_PCA.fit(X_train_pca, y_train)

train_pred = model_SVM_PCA.predict(X_train_pca)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(SVM - pca) of training set: {accuracy}")
print(f"Precision(SVM - pca) of training set: {precision}")
print(f"F1-score (SVM - pca) of training set: {f1}")

test_pred = model_SVM_PCA.predict(X_test_pca)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(SVM - pca) of test set: {accuracy}")
print(f"Precision(SVM - pca) of test set: {precision}")
print(f"F1-score (SVM - pca) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))
print("----------------SVM-----------------------")
print("\n\n\n")


#MLP
#Search for best parameters
model = MLPClassifier()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
parameters = {
    'activation': ['logistic', 'relu'],
    'solver': ['lbfgs', 'adam'],
    'max_iter': [100, 200, 300]
}
clf = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=kfold, refit=True, verbose=3)
clf.fit(X_train_std, y_train)

print("Searcing for best parameters... -MLP-")
print(clf.best_score_)
print(clf.best_params_)


#Basic
print("----------------MLP-----------------------")
print("\n\n\n")
model_MLP_basic = MLPClassifier(alpha=1e-3, activation='relu', hidden_layer_sizes=(7),max_iter=300, solver='adam', random_state=1, early_stopping=True)
model_MLP_basic.fit(X_train_std, y_train)

train_pred = model_MLP_basic.predict(X_train_std)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(MLP - basic) of training set: {accuracy}")
print(f"Precision(MLP - basic) of training set: {precision}")
print(f"F1-score (MLP - basic) of training set: {f1}")

test_pred = model_MLP_basic.predict(X_test_std)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(MLP - basic) of test set: {accuracy}")
print(f"Precision(MLP - basic) of test set: {precision}")
print(f"F1-score (MLP - basic) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))


#With LDA
model_MLP_lda = MLPClassifier(alpha=1e-3, activation='relu', hidden_layer_sizes=(7),max_iter=300, solver='adam', random_state=1, early_stopping=True)
model_MLP_lda.fit(X_train_lda, y_train)

train_pred = model_MLP_lda.predict(X_train_lda)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(MLP - lda) of training set: {accuracy}")
print(f"Precision(MLP - lda) of training set: {precision}")
print(f"F1-score (MLP - lda) of training set: {f1}")

test_pred = model_MLP_lda.predict(X_test_lda)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(MLP - lda) of test set: {accuracy}")
print(f"Precision(MLP - lda) of test set: {precision}")
print(f"F1-score (MLP - lda) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))
print("----------------MLP-----------------------")
print("\n\n\n")


#With PCA
model_MLP_pca = MLPClassifier(alpha=1e-3, activation='relu', hidden_layer_sizes=(7),max_iter=300, solver='adam', random_state=1, early_stopping=True)
model_MLP_pca.fit(X_train_pca, y_train)

train_pred = model_MLP_pca.predict(X_train_pca)
accuracy = accuracy_score(y_train, train_pred)
precision = precision_score(y_train, train_pred, average='macro')
f1 = f1_score(y_train, train_pred, average='macro')
print(f"Accuracy(MLP - pca) of training set: {accuracy}")
print(f"Precision(MLP - pca) of training set: {precision}")
print(f"F1-score (MLP - pca) of training set: {f1}")

test_pred = model_MLP_pca.predict(X_test_pca)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average='macro')
f1 = f1_score(y_test, test_pred, average='macro')
print(f"Accuracy(MLP - pca) of test set: {accuracy}")
print(f"Precision(MLP - pca) of test set: {precision}")
print(f"F1-score (MLP - pca) of test set: {f1}")

cm = confusion_matrix(y_train, train_pred)
print("LJUTNJA      NEUTRALNO       RADOST      STRAH       TUGA")
print(cm.diagonal()/cm.sum(axis=1))
print("----------------MLP-----------------------")
print("\n\n\n")
