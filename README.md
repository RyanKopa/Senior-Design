# Senior Design
EMS 188A/B

This script will take data recorded from mechanical tests measuring the strength of gold ribbon bonding.  Data will be saved in a csv file with the following format:

sampleID,temperature,bondDuration,voltagePower,bondForce,bondMiddle,bondedArea,bondItself,labels

The features of the data are everything but the labels, which are aptly the labels of the data set.  Labels are the failure mode of the bonds post-mechanical testing.

GBM (XGBoost) is used to predict the failure modes of the bonds based on their bonding parameters.  Ideally, we are looking for heel break failure modes.