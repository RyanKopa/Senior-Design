
# coding: utf-8

# ### Model that will predict strength of bond based on bonding parameters
# Currently data is read from a csv file with the columns: 
# **sampleID,temperature,bondDuration,voltagePower,bondForce,**
# **bondMiddle,bondedArea,bondItself,labels,heelDeformation,**
# **centerDeformation,failureMode**
# 
# * sampleID - substrate number
# * temperature - temperature at which bonds are made, in Celsius 
# * bond duration - measured in milliseconds, duration of bonding
# * voltage power - measured in volts, power used on bonds
# * bond force - measured in gram force, force used on bonds
# * bond middle - length (long) of deformation area of the ribbon
# * bond area - length (long) of heat infected area of the ribbon
# * bond itself - length of ribbon that isn't deformed
# * bond length difference - difference in length between the deformation area and undeformed ribbon
# * labels - strength measurements, in grams, of the bonds
# * heel deformation - percent change measurements of the deformation area of ribbon compared to undeformed ribbon
# * center deformation - percent change measurements of the heat infected area of the ribbon compared to undeformed ribbon
# * failure mode - noted failure mode of the ribbon during mechanical testing
# * annealed - 0 or 1, noting if bond was not annealed or annealed
# * annealing temperature - note of what the annealing temperature for the bond was, in Kelvin
# * annealing duration - note of how long the bond was annealed for, in seconds
# 
# Two models are run, where labels and failure mode are the labels, and the rest of the data (minus the sampleID) are the features.  Feature important plots and data mapping plots are included as well.

# In[1]:

import time
import pandas as pd
from sklearn.cross_validation import train_test_split #chang sklearn.cross_validation to sklearn.model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb


# In[2]:

dfTrain = pd.read_csv('GBM Analysis - anneal.csv')
labels = dfTrain['labels'].values


# In[3]:

dfTrain.info()


# In[4]:

le = LabelEncoder()
dfTrain['failureMode'] = le.fit_transform(dfTrain['failureMode'])
labelsFailureMode = dfTrain['failureMode'].values


# Split the training and validating data for the model

# In[5]:

trainData, validateData = train_test_split(dfTrain, test_size=0.05,
                                           random_state=42)


# In[6]:

trainLabels = trainData['labels'].values
validateLabels = validateData['labels'].values

trainLabelsFailureMode = trainData['failureMode'].values
validateLabelsFailureMode = validateData['failureMode'].values

# dfTrain['labels'] = dfTrain['labels'].astype(float)

drop_var = ['sampleID', 'labels', 'failureMode']

# Drop id column, and labels that are added later
trainData = trainData.drop(drop_var, axis=1)
validateData = validateData.drop(drop_var, axis=1)

# These data matrices (trainDMatrix and validateDMatrix) are for the strength predictions 
trainDMatrix = xgb.DMatrix(trainData.as_matrix(),
                           label=trainLabels.astype(int))

validateDMatrix = xgb.DMatrix(validateData.as_matrix(),
                              label=validateLabels.astype(int))

# These data matrices (trainDMatrixFailure and validateDMatrixFailure) are for the failure mode predictions 
trainDMatrixFailure = xgb.DMatrix(trainData.as_matrix(),
                           label=trainLabelsFailureMode.astype(int))

validateDMatrixFailure = xgb.DMatrix(validateData.as_matrix(),
                           label=validateLabelsFailureMode.astype(int))


# In[7]:

validateData.info()


# # Strength Prediction testing
# Currently this model for predicting bond strength based on bonding and deformation parameters runs at 95% accuracy.

# In[8]:

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


# In[9]:

watchlist = [(trainDMatrix, 'train'), (trainDMatrix, 'eval')]


# In[10]:

start_time = time.time()
clf = xgb.train(params, trainDMatrix, 210, evals=watchlist,
                early_stopping_rounds=60, verbose_eval=True)
print('Time taken to classify')
print((time.time() - start_time)/60)


# ##### merror is the percent of incorrect cases, here we have a 95% accuracy at predicting strength of bonds

# ### Validation testing
# Currently in progress to develop a new accuracy testing function

# In[11]:

# validate_label_predictions = clf.predict(validateDMatrix)
# print(accuracy_score(validateLabels, validate_label_predictions.argmax(axis = 1)))


# ### Feature Importance Mapping

# In[12]:

import operator
importances = clf.get_fscore()
print(importances)


# In[93]:

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
# Currently, this model does not work well (90% inaccuracy) due to too much of the data consisting of heel break failure modes.  The model needs to have more data that consists of different failure modes.

# In[14]:

watchlist2 = [(trainDMatrixFailure, 'train'), (validateDMatrixFailure, 'eval')]


# In[15]:

start_time2 = time.time()
clf2 = xgb.train(params, trainDMatrixFailure, 210, evals=watchlist,
                early_stopping_rounds=50, verbose_eval=True)
print('Time taken to classify')
print((time.time() - start_time2)/60)


# In[16]:

import operator
importances = clf2.get_fscore()
print(importances)


# In[17]:

import matplotlib.pyplot as plt
# xgb.plot_importance(model)
# plt.show()
clf2.get_fscore()
mapper = {'f{0}'.format(i): v for i, v in enumerate(trainData.columns)}
mapped = {mapper[k]: v for k, v in clf.get_fscore().items()}
mapped
xgb.plot_importance(mapped, color='red')
plt.show()


# In[18]:

dfTrain.info()


# # Data Visualization of Failure Modes
# Data visualization of the parameters of bonding (bond duration, voltage power, and bond force) mapped to the failure mode.

# In[87]:

colormap = dfTrain[['failureMode', 'annealed']].copy()
colormap.loc[colormap['failureMode'] == 0, 'failureMode'] = 'r' # foot lift
colormap.loc[colormap['failureMode'] == 1, 'failureMode'] = 'g' # Heel Break
colormap.loc[colormap['failureMode'] == 2, 'failureMode'] = 'b' # Ribbon Break
colormap.loc[colormap['annealed'] == 1, 'annealed'] = 100
colormap.loc[colormap['annealed'] == 0, 'annealed'] = 200


# In[88]:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
bondDuration = dfTrain['bondDuration'].values
voltagePower = dfTrain['voltagePower'].values
bondForce = dfTrain['bondForce'].values
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(bondDuration, voltagePower, bondForce, s=colormap.annealed.values.tolist(), 
           color=colormap.failureMode.values.tolist())
ax.set_xlabel('Bond Duration (ms)')
ax.set_ylabel('Voltage Power (V)')
ax.set_zlabel('Bond Force (gf)')
plt.show()
print('red is foot lift, blue is heel break, green is heel break')


# # Data Visualization of Strength
# Data visualization of the parameters of bonding (bond duration, voltage power, and bond force) mapped to the strengths of bonds. 

# In[89]:

colormapStrengths = pd.DataFrame(labels, columns = ['strengths'])
colormapStrengths['colors'] = np.zeros(len(colormapStrengths.strengths))
colormapStrengths['sizes'] = dfTrain.annealed.copy()


# In[91]:

colormapStrengths.loc[colormapStrengths['strengths'] < 6, 'colors'] = 'r'
colormapStrengths.loc[colormapStrengths['strengths'] < 3, 'colors'] = 'g'
colormapStrengths.loc[colormapStrengths['strengths'] > 6, 'colors'] = 'b'
colormapStrengths.loc[colormapStrengths['sizes'] == 1, 'sizes'] = 100
colormapStrengths.loc[colormapStrengths['sizes'] == 0, 'sizes'] = 200


# In[40]:

# # # Old Colormap Scheme using six colors, given up on because it was too confusing
# # new column of dataframe is written over based on the color correlating to the minimum strength
# colormapStrengths.loc[colormapStrengths['strengths'] < 10, 'colors'] = 'r'
# colormapStrengths.loc[colormapStrengths['strengths'] < 5, 'colors'] = 'gold'
# colormapStrengths.loc[colormapStrengths['strengths'] < 4, 'colors'] = 'dodgerblue'
# colormapStrengths.loc[colormapStrengths['strengths'] < 3, 'colors'] = 'g'
# colormapStrengths.loc[colormapStrengths['strengths'] < 2, 'colors'] = 'm'
# colormapStrengths.loc[colormapStrengths['strengths'] < 1, 'colors'] = 'indigo'
# colormapStrengths.loc[colormapStrengths['strengths'] < 0, 'colors'] = 'k'


# In[92]:

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(bondDuration, voltagePower, bondForce, s=colormapStrengths.sizes.values.tolist(), 
           color=colormapStrengths.colors.values.tolist())
ax.set_xlabel('Bond Duration (ms)')
ax.set_ylabel('Voltage Power (V)')
ax.set_zlabel('Bond Force (gf)')
plt.show()
print('black is less than 3 grams')
print('red is between 3 and 6 grams')
print('blue is greater than 6 grams')


# In[ ]:




# In[ ]:




# In[ ]:



