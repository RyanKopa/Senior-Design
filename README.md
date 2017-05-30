# Senior Design
## University of California, Davis EMS 188A/B

This script will take data recorded from mechanical tests measuring the strength of gold ribbon bonding.  Data will be saved in a csv file with the following format:

sampleID,temperature,bondDuration,voltagePower,bondForce,bondMiddle,bondedArea,bondItself,labels,heelDeformation,centerDeformation,failureMode

GBM (XGBoost) is used to predict the failure modes of the bonds based on their bonding parameters.  Ideally, we are looking for heel break failure modes and strengths (labels) of greater than 6 grams force.
