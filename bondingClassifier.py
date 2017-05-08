#Model to predict bonding parameters on failure mode

import time
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb

# pylint: disable=C0103

dfTrain = pd.read_csv('')#to be filled in later
labels = dfTrain[''].values

### Feature Engineering here if necessary ###

trainData, validateData = train_test_split(dfTrain, test_size=0.1,
                                           random_state=42)

le = LabelEncoder()
trainLabels = le.fit_transform(trainData['labels'].values)
validateLabels = le.fit_transform(validateData['labels'].values)

drop_var = ['id', 'labels']

trainData = trainData.drop(drop_var, axis=1)
validateData = validateData.drop(drop_var, axis=1)

trainDMatrix = xgb.DMatrix(trainData.as_matrix(),
                           label=trainLabels.astype(int))

validateDMatrix = xgb.DMatrix(validateData.as_matrix(),
                              label=validateLabels.astype(int))

# Training Classifier for parameter tuning

start_time = time.time()
watchlist = [(validateDMatrix, 'eval'), (trainDMatrix, 'train')]

params = {
    'learning_rate'    : 0.038,
    'colsample_bytree' : 0.6,
    'subsample'        : 0.65,
    'max_depth'        : 7,
    'num_class'        : len(np.unique(trainLabels)),
    'seed'             : 0,
    'objective'        : 'multi:softprob',
    'eval_metric'      : 'mlogloss',
    'booster'          : 'gbtree'
}

clf = xgb.train(params, trainDMatrix, 330, evals=watchlist,
                early_stopping_rounds=10, verbose_eval=True)

print((time.time() - start_time)/60)
validate_label_predictions = clf.predict(validateDMatrix)
print(accuracy_score(validateLabels, validate_label_predictions))
