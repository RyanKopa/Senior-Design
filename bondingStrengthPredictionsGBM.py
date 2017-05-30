
# coding: utf-8

# ### Model that will predict strength of bond based on bonding parameters

# In[2]:

import time
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb


# In[3]:

dfTrain = pd.read_csv('GBM Analysis - Sheet1.csv')
labels = dfTrain['labels'].values


# In[4]:

dfTrain.info()


# In[5]:

le = LabelEncoder()
dfTrain['failureMode'] = le.fit_transform(dfTrain['failureMode'])
labelsFailureMode = dfTrain['failureMode'].values


# In[6]:

trainData, validateData = train_test_split(dfTrain, test_size=0.05,
                                           random_state=42)


# In[7]:

trainLabels = trainData['labels'].values
validateLabels = validateData['labels'].values

trainLabelsFailureMode = trainData['failureMode'].values
validateLabelsFailureMode = validateData['failureMode'].values

# dfTrain['labels'] = dfTrain['labels'].astype(float)

drop_var = ['sampleID', 'labels', 'failureMode']

trainData = trainData.drop(drop_var, axis=1)
validateData = validateData.drop(drop_var, axis=1)

trainDMatrix = xgb.DMatrix(trainData.as_matrix(),
                           label=trainLabels.astype(int))

validateDMatrix = xgb.DMatrix(validateData.as_matrix(),
                              label=validateLabels.astype(int))

trainDMatrixFailure = xgb.DMatrix(trainData.as_matrix(),
                           label=trainLabelsFailureMode.astype(int))

validateDMatrixFailure = xgb.DMatrix(validateData.as_matrix(),
                           label=validateLabelsFailureMode.astype(int))


# In[8]:

validateData.info()


# # Strength Prediction testing
# 
# ## Training Classifier for parameter tuning

# In[9]:

params = {
    'learning_rate'    : 0.038,
    'colsample_bytree' : 0.6,
    'subsample'        : 0.65,
    'max_depth'        : 7,
    'num_class'        : len(np.unique(trainLabels)),
    'seed'             : 0,
    'objective'        : 'multi:softprob',
    'eval_metric'      : 'merror',
    'booster'          : 'gbtree'
}


# In[10]:

watchlist = [(trainDMatrix, 'train'), (trainDMatrix, 'eval')]


# In[11]:

start_time = time.time()
clf = xgb.train(params, trainDMatrix, 210, evals=watchlist,
                early_stopping_rounds=60, verbose_eval=True)
print('Time taken to classify')
print((time.time() - start_time)/60)


# ### Validation testing

# In[12]:

# validate_label_predictions = clf.predict(validateDMatrix)
# print(accuracy_score(validateLabels, validate_label_predictions.argmax(axis = 1)))


# ### Feature Importance Mapping

# In[13]:

import operator
importances = clf.get_fscore()
print(importances)


# In[14]:

import matplotlib.pyplot as plt
# xgb.plot_importance(model)
# plt.show()
clf.get_fscore()
mapper = {'f{0}'.format(i): v for i, v in enumerate(trainData.columns)}
mapped = {mapper[k]: v for k, v in clf.get_fscore().items()}
mapped
xgb.plot_importance(mapped, color='red')
plt.show()


# # Failure Mode Prediction Testing
# 
# ## Training Classifier for parameter tuning

# In[15]:

watchlist2 = [(trainDMatrixFailure, 'train'), (validateDMatrixFailure, 'eval')]


# In[16]:

start_time2 = time.time()
clf2 = xgb.train(params, trainDMatrixFailure, 210, evals=watchlist,
                early_stopping_rounds=50, verbose_eval=True)
print('Time taken to classify')
print((time.time() - start_time2)/60)


# In[17]:

import operator
importances = clf2.get_fscore()
print(importances)


# In[18]:

import matplotlib.pyplot as plt
# xgb.plot_importance(model)
# plt.show()
clf2.get_fscore()
mapper = {'f{0}'.format(i): v for i, v in enumerate(trainData.columns)}
mapped = {mapper[k]: v for k, v in clf.get_fscore().items()}
mapped
xgb.plot_importance(mapped, color='red')
plt.show()


# In[19]:

dfTrain.info()


# In[20]:

# from sklearn.cluster import KMeans
# # bondDuration = dfTrain['bondDuration'].values
# # voltagePower = dfTrain['voltagePower'].values
# # bondForce = dfTrain['bondForce'].values
# # strengthLabels = dfTrain.labels.values
# data = dfTrain[['bondDuration','voltagePower','bondForce']].values
# kmeans = KMeans(n_clusters=3).fit(data)


# In[21]:

# kmeans


# In[22]:

# print(kmeans.labels_)


# In[23]:

# print(dfTrain.failureMode.values)


# In[24]:

colormap = dfTrain[['failureMode']].copy()
colormap.loc[colormap['failureMode'] == 0, 'failureMode'] = 'r' # foot lift
colormap.loc[colormap['failureMode'] == 1, 'failureMode'] = 'g' # Heel Break
colormap.loc[colormap['failureMode'] == 2, 'failureMode'] = 'b' # Ribbon Break
print(colormap.failureMode.values)


# In[25]:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
bondDuration = dfTrain['bondDuration'].values
voltagePower = dfTrain['voltagePower'].values
bondForce = dfTrain['bondForce'].values
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(bondDuration, voltagePower, bondForce, color=colormap.failureMode.values.tolist())
ax.set_xlabel('Bond Duration')
ax.set_ylabel('Voltage Power')
ax.set_zlabel('Bond Force')
plt.show()
print('red is foot lift, blue is heel break, green is heel break')


# In[28]:

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# bondDuration = dfTrain['bondDuration'].values
# voltagePower = dfTrain['voltagePower'].values
# bondForce = dfTrain['bondForce'].values
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(bondDuration, voltagePower, bondForce, color=colormap.failureMode.values.tolist())
# ax.set_xlabel('Bond Duration')
# ax.set_ylabel('Voltage Power')
# ax.set_zlabel('Bond Force')
# plt.show()
# print('red is foot lift, blue is heel break, green is heel break')


# In[60]:

colormapStrengths = pd.DataFrame(labels, columns = ['strengths'])
colormapStrengths['colors'] = np.zeros(len(colormapStrengths.strengths))


# In[68]:

colormapStrengths.loc[colormapStrengths['strengths'] < 10, 'colors'] = 'r'
colormapStrengths.loc[colormapStrengths['strengths'] < 5, 'colors'] = 'gold'
colormapStrengths.loc[colormapStrengths['strengths'] < 4, 'colors'] = 'dodgerblue'
colormapStrengths.loc[colormapStrengths['strengths'] < 3, 'colors'] = 'g'
colormapStrengths.loc[colormapStrengths['strengths'] < 2, 'colors'] = 'm'
colormapStrengths.loc[colormapStrengths['strengths'] < 1, 'colors'] = 'indigo'
colormapStrengths.loc[colormapStrengths['strengths'] < 0, 'colors'] = 'k'


# In[70]:

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(bondDuration, voltagePower, bondForce, color=colormapStrengths.colors.values.tolist())
ax.set_xlabel('Bond Duration')
ax.set_ylabel('Voltage Power')
ax.set_zlabel('Bond Force')
plt.show()
print('red is greater than 6')
print('gold is between 5 and 6')
print('light blue is between 4 and 5')
print('green is between 3 and 4')
print('violet is between 2 and 3')
print('indigo is between 1 and 2')
print('black is less than 1')


# In[ ]:



