<h1 align="center">DEEP ENSEMBLE</h1>

## Description
<p>In the context of deep learning, This library is a wrapper, It allows one to perform ensembling techniques such as Stacked Ensembling, Weighted Ensembling, Ensembling based on Votes.
 This library get in an array of predictions made by different models on which the user wants to perform ensembling and it performs either of the three ensembling technique based on user choice.

- What this does?
    * Gets array of predictions(probabilities)
    * Performs ensembling
    * Returns Score of the metrics(user prefers)
    * Returns Y_predicted resulting due to ensembling
    
- Ensembling Techniques included
    * Weighted Ensembling
    * Ensembling by Voting
    * Stacked Ensembling
    
</p>

## Deep Ensemble

<p>Deep Ensemble is a python package for ensembling the prediciton results. </p>

## How to use:

Step 1:
  Install the libaray

````python
pip install DeepEnsemble
````
Step 2:

  Import the library, and specify the path of the csv file. 
````python
from DeepEnsemble.DeepEnsemble import DeepEnsembler

Ensembler = DeepEnsembler(Y_pred, Y_actual, type=None, predThreshold=0.5, metrics="accuracy_score")
score,Y_pred_ensembled = Ensembler.WeightedEnsembling()

````
Note:
  * Y_pred   - Array of predictions made by models
  * Y_actual - Array of actual Class 
  
  There are some optional parameters that you can specify as listed below,

## Usage:

````python
from DeepEnsemble.DeepEnsemble import DeepEnsembler
Y_Pred = np.array([results of model1],
                  [results of model2],
                  [results of model3],
                   ....)
Y_actual = np.array([actual class])

---Example1:
Ensembler = DeepEnsembler(Y_Pred, Y_actual, type="Weighted", predThreshold=0.6, metrics="cohen_kappa_score")
score,Y_pred_ensembled = Ensembler.WeightedClassifier()

---Example2:
Ensembler = DeepEnsembler(Y_Pred, Y_actual, type="Voting", predThreshold=0.6, metrics="cohen_kappa_score")
score,Y_pred_ensembled = Ensembler.VotingClassifier()

---Example3:
Ensembler = DeepEnsembler(Y_Pred, Y_actual, type="Stacking", predThreshold=0.6, metrics="cohen_kappa_score")
score,Y_pred_ensembled = Ensembler.StackingClassifier()
````

## Parameters

------

| Parameter | Default Value | Limit | Example |
| ------ | ------ | ------ | ------ |
| Y_pred | ***none*** | Provide a array with results of different models. | np.array([results of model1],[results of model2],[results of model3],....) |
| Y_actual | ***,*** | Provide a array with actual class | np.array([actual class])
| type | ***Weighted*** | Weighted, Voted, Stacked |  | 
| predThreshold | ***0.5*** | 0 to 1 | 0.5 | 
| metrics | Sklearn Classification metrics | specify any valid classificaiton metrics from sklearn | "accuracy_score" | 



<h2 align="center"> --- THANK YOU, CHEERS --- </h2>
