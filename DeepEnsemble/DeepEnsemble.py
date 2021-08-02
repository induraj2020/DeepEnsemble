import tensorflow as tf
import numpy as np
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

metrics_list = ["cohen_kappa_score", "roc_auc_score", "top_k_accuracy_score","accuracy_score","balanced_accuracy",
                "average_precision","neg_brier_score","f1","f1_macro","f1_macro","f1_weighted","f1_samples",
                "neg_log_loss","precision","recall","jaccard"]

class DeepEnsembler():
    def __init__(self, Y_pred, Y_actual, type="Weighted", predThreshold=0.5, metrics="accuracy_score"):
        self.X, self.Y = self.checkInstance(Y_pred,Y_actual)
        self.predThreshold = self.checkPredThreshol(predThreshold)
        self.type          = self.checkType(type)
        if metrics in metrics_list:
            self.metrics       = metrics
        else:
            self.metrics       = metrics

    def checkInstance(self, x, y):
        if isinstance(x, (np.ndarray, np.generic)) and isinstance(y, (np.ndarray, np.generic)):
            assert x.shape[0] == y.shape[0], 'shapes of Y_pred and Y_actual does not match'
            return x, y
        else:
            print("Inputs Y_pred and Y_Actual must be Arrays")

    def checkPredThreshol(self, threshold):
        assert 0<=threshold<=1, "predThreshold must be between 0 and 1"
        return threshold

    def checkType(self, ensembletype):
        if ensembletype==None:
            ensembletype="Weighted"
            return ensembletype
        elif ensembletype in ["Stacking", "Weighted", "Voting"]:
            return ensembletype
        else:
            ensembletype = "Weighted"
            return ensembletype

    def WeightedClassifier(self):
        if self.type=="Weighted":
            print("Weighted Ensembling.....")
            ModelList = self.X.shape[1]
            temp_ensemble_score = float("-inf")
            for model_ouput in range(self.X.shape[1]):
                globals()[f'y_pred_{model_ouput}'] = self.X[:, model_ouput]
            for j in range(10000):
                random_weight = np.random.dirichlet(np.ones(ModelList), size=1)[0]
                pred_prob = np.sum(np.array([globals()[f'y_pred_{i}'] * random_weight[i] for i in range(ModelList)]),axis=0)
                pred_prob_round = np.where(pred_prob > self.predThreshold, 1, 0)
                ensemble_score = eval(self.metrics)(self.Y, pred_prob_round)
                max_flag = ensemble_score > temp_ensemble_score
                if max_flag:
                    best_ensemble_score = ensemble_score
                    best_weight = random_weight
                    final_class_prob = pred_prob
                    final_class  = pred_prob_round
            print("Final best ensemble score found is {}".format(best_ensemble_score))
            return best_ensemble_score, final_class
        else:
            print("Type of ensembling choosen during class intialization is different, \n"
                  "if you used type=\"weighted\" call  WeightedEnsembling method",
                  "if you used type=\"Voted\" call  VotingClassifier method",
                  "if you used type=\"Voted\" call  StackingClassifier method",)

    def VotingClassifier(self):
        if self.type=="Voting":
            y_pred = np.where(self.X > self.predThreshold, 1, 0)
            y_pred_count = np.count_nonzero(y_pred, axis=1)
            final_array = np.zeros(self.X.shape[0])
            numb_zeros = y_pred.shape[1] - np.count_nonzero(y_pred, axis=1)
            for i in range(final_array.shape[0]):
                if numb_zeros[i] > y_pred_count[i]:
                    final_array[i] = 0
                elif numb_zeros[i] < y_pred_count[i]:
                    final_array[i] = 1
                else:
                    final_array[i] = np.random.choice([0, 1])
            voted_score = eval(self.metrics)(final_array, self.Y)
            print("Best Voted score achieved {}".format(voted_score))
            return voted_score, final_array
        else:
            print("Type of ensembling choosen during class intialization is different, \n"
                  "if you used type=\"weighted\" call  WeightedEnsembling method",
                  "if you used type=\"Voted\" call  VotingClassifier method",
                  "if you used type=\"Voted\" call  StackingClassifier method", )
    def StackingClassifier(self):
        if self.type=="Stacking":
            try:
                n_estimators = [50,100,500,800,1500,2500,5000]
                max_features = ["auto","sqrt","log2"]
                max_depth = [10,20,30,40,50]
                max_depth.append(None)
                min_samples_split = [1,2,5,10,15,20]
                min_samples_leaf = [1,2,5,10,15]
                grid_param = {"n_estimators":n_estimators,
                              "max_features":max_features,
                              'max_depth':max_depth,
                              'min_samples_split':min_samples_split,
                              'min_samples_leaf':min_samples_leaf}
                RF = RandomForestClassifier()
                RFR_Random= RandomizedSearchCV(estimator=RF,
                                               param_distributions=grid_param,
                                               n_iter=500,
                                               cv=5, verbose=0, random_state=42, n_jobs=-1)
                y_pred = np.where(self.X > self.predThreshold, 1, 0)

                X_train, X_test, y_train, y_test = train_test_split(y_pred, self.Y, test_size=0.3)  # 70% training and 30% test
                RFR_Random.fit(X_train, y_train)
                best_model = RFR_Random.best_estimator_
                y_pred_stacked = best_model.predict(X_test)
                stacked_score = eval(self.metrics)(y_pred_stacked, y_test)
                print("Best stacked score achieved {}".format(stacked_score))
                return stacked_score, y_test
            except ValueError as e:
                print("Insufficient number of samples")
                print(e)
        else:
            print("Type of ensembling choosen during class intialization is different, \n"
                  "if you used type=\"weighted\" call  WeightedEnsembling method",
                  "if you used type=\"Voted\" call  VotingClassifier method",
                  "if you used type=\"Voted\" call  StackingClassifier method", )

if __name__ =="__main__":
    xx = np.array([[0.1, 0.2, 0.3, 0.4],
                   [0.5, 0.6, 0.7, 0.8],
                   [0.8, 0.6, 0.3, 0.4],
                  [0.8, 0.6, 0.7, 0.8],
                  [0.8, 0.8, 0.8, 0.4],
                  [0.1, 0.6, 0.7, 0.8],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.5, 0.6, 0.7, 0.8],
                   [0.8, 0.6, 0.3, 0.4],
                   [0.8, 0.6, 0.7, 0.8],
                   [0.8, 0.8, 0.8, 0.4],
                   [0.1, 0.6, 0.7, 0.8]
                   ])
    y_actual = np.array([1,0,0,1,0,1,1,0,0,1,0,1])
    ensembler = DeepEnsembler(xx, y_actual, type='Voting')
    _,_= ensembler.VotingClassifier()