import pandas as pd 
import numpy as np
import gzip
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

def scale(X_orig):
    nom = (X_orig-X_orig.min(axis=0))
    print("shape of nom",np.shape(nom))
    denom = (X_orig.max(axis=1)-X_orig.min(axis=1))
    denom[denom==0] = 1
    nom[np.reshape(denom,(45,))==0]=0.5
    ret_val=np.transpose(nom)/denom
    return np.transpose(ret_val) 


dataframe = pd.read_csv("readmission.csv")  
dataframe.replace('?',np.NaN,inplace=True)
dataframe.nunique()
selected_encounters = dataframe.groupby('patient_nbr', as_index=False)['encounter_id'].idxmin()
curated_df = dataframe.loc[selected_encounters]
use_dataframe=curated_df


def shuffle(df):
    return df.sample(frac=1).reset_index(drop=True)

# drop columns with specified label
def dropColumn(df, colLabels):
    for colLabel in colLabels:
        if colLabel in df.columns:
            df.drop([colLabel], axis=1, inplace=True)
    return df

# convert feature from categorical to binary
def categoricalToBinary(df, colLabel):
    if colLabel in df.columns:
        df = pandas.get_dummies(df, columns=[colLabel])
    return df

# return features (X) and target (Y) from dataframe
def seperateDataTargets(df, targetLabel):
    X = df.drop(targetLabel, axis=1).values
    Y = df[targetLabel].values.ravel()
    return X, Y

# convert to binary classes: 1 (positive) or 0 (negative)
def convertToBinaryClass(y):
    for i in range(len(y)):
        if y[i] != 1:
            y[i] = 0
        else:
            y[i] = 1
    return np.array(y)

target_name = 'readmitted'
use_dataframe = shuffle(use_dataframe)
use_dataframe = dropColumn(use_dataframe, ['encounter_id', 'patient_nbr', 'weight','payer_code', 'medical_specialty'])

obj_to_category_list = use_dataframe.select_dtypes(include=['object']).columns.tolist()

for obj in obj_to_category_list:
    use_dataframe[obj] = use_dataframe[obj].astype('category')


# save label encodings to le_dict
le_dict = {}
cols_for_le = obj_to_category_list

for col in cols_for_le:
    le_dict[col] = dict(enumerate(use_dataframe[col].cat.categories))


# label encoding
for col in cols_for_le:
    use_dataframe[col] = use_dataframe[col].cat.codes

cols_for_he = [ 'diag_1', 'diag_2', 'diag_3','race' , 'gender', 'age' ,'glyburide-metformin','insulin','miglitol','acarbose','rosiglitazone','pioglitazone','glyburide','glipizide','glimepiride','chlorpropamide','nateglinide','repaglinide','metformin','A1Cresult','max_glu_serum'] # can try: 'diag_1', 'diag_2', 'diag_3'
use_dataframe = pd.get_dummies(use_dataframe, columns=cols_for_he, dummy_na=True)

# one hot encoding
# cols_for_he = ['RACE','ETHNICITY','Molecular_Profile','Mutation','Drugs','Type'] # removed 'BIRTHPLACE'
# use_dataframe = pandas.get_dummies(use_dataframe, columns=cols_for_he, dummy_na=True)


# convert boolean classes to int (0 - False, 1 - True)
# use_dataframe['Diabetic'] = use_dataframe['Diabetic'].astype(int)
# use_dataframe['Hypertension'] = use_dataframe['Hypertension'].astype(int)


# seperate X, Y and binarize outcome Y
X, Y = seperateDataTargets(use_dataframe, [target_name])

skiplabels = ['diag_1', 'diag_2', 'diag_3']
keys = list(le_dict.keys())
#pp.pprint([{k: le_dict[k]} for k in keys if k not in skiplabels])


feature_names = use_dataframe.drop([target_name, 'encounter_id', 'patient_nbr','weight','payer_code'], axis=1, errors='ignore').columns.tolist()
class_names = ['>30', '<30', 'NO'] # >30 readmission, <30 days readmission, No readmission

# number of rows and columns
num_cols = use_dataframe.shape[1]
num_rows = use_dataframe.shape[0]

# print basic data set characteristics
print('\n'.join(feature_names))
print(use_dataframe.head())
label_dataframe = use_dataframe[['readmitted']].copy()
del use_dataframe['readmitted']
use_dataframe=use_dataframe.as_matrix()
use_dataframe=np.transpose(use_dataframe)
rows=np.shape(use_dataframe)[0]
cols=np.shape(use_dataframe)[1]
data_y=label_dataframe.as_matrix()
print("shape of data_y using labels", np.shape(data_y))
print("leftover features",np.shape(use_dataframe)[0])
data_y=np.reshape(data_y,(1,cols))
print(data_y)
#data_y[data_y < 2] = 1
print(data_y)
print("shape of data", np.shape(use_dataframe))
print("shape of data_y", np.shape(data_y))
trans_use_dataframe=np.transpose(use_dataframe)

nm = NearMiss(sampling_strategy='majority', version=2)
X_res, y_res = nm.fit_resample(trans_use_dataframe, np.transpose(data_y).ravel())
print("Shape of resampled data", np.shape(X_res))
print("Shape of resampled labels", np.shape(y_res))

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_res, y_res)
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
conf_mat = confusion_matrix(y_res, y_pred)
print(conf_mat)

#scores = cross_val_score(clf,X_res, y_res, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
