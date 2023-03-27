from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb


NUMESTIMATORS=100
class RandomForestManager:
    def __init__(self, data, dropcols=[], outcomevar="tree_name", numestimators=NUMESTIMATORS):
        self.data = data
        self.dropcols = dropcols
        self.outcomevar = outcomevar
        self.numestimators = numestimators
        pass


    def LOOCV_featureselection(self, data, ids, outcomevar, dropcols, idcolumn, numestimators=NUMESTIMATORS):
        """
            Intermediate function.
        """
        # Separate data for leave-one-person-out-cross-validation (LOOCV)
        LOOCV_O = ids
        data[idcolumn] = data[idcolumn].apply(str)
        data_filtered = data[data[idcolumn] != LOOCV_O]
        data_cv = data[data[idcolumn] == LOOCV_O]

        # Train data - all other people in dataframe
        data_train = data_filtered.drop(columns=dropcols)
        X_train = data_train.drop(columns=[outcomevar])

        feature_list = list(X_train.columns)
        X_train = np.array(X_train)
        y_train = np.array(data_train[outcomevar])  # Outcome variable here

        from sklearn.ensemble import RandomForestRegressor
        # Instantiate model with numestimators decision trees
        rf = RandomForestClassifier(n_estimators=numestimators, random_state=0)
        # Train the model on training data
        rf.fit(X_train, y_train);

        # Get importances:
        importances = list(rf.feature_importances_)  # List of tuples with variable and importance
        important = pd.DataFrame()
        important['value'] = feature_list
        important['importances'] = importances

        return important
    def RFLOOCV(self, data, ids, outcomevar, dropcols, idcolumn, numestimators=NUMESTIMATORS, fs=0.05):
        """
            Intermediate function.

        """

        # Get important features
        listimportances = self.LOOCV_featureselection(data, ids, outcomevar, dropcols, idcolumn, numestimators)
        filteredi = listimportances[listimportances['importances'] < fs]
        filteredi = filteredi['value']

        LOOCV_O = str(ids)
        data[idcolumn] = data[idcolumn].apply(str)
        data_filtered = data[data[idcolumn] != LOOCV_O]
        data_cv = data[data[idcolumn] == LOOCV_O]

        # Test data - the person left out of training
        data_test = data_cv.drop(columns=dropcols)
        data_test = data_test.drop(columns=filteredi)  # cvf
        X_test = data_test.drop(columns=[outcomevar])
        y_test = data_test[outcomevar]  # This is the outcome variable

        # Train data - all other people in dataframe
        data_train = data_filtered.drop(columns=dropcols)
        data_train = data_train.drop(columns=filteredi)
        X_train = data_train.drop(columns=[outcomevar])

        feature_list = list(X_train.columns)
        X_train = np.array(X_train)
        y_train = np.array(data_train[outcomevar])  # Outcome variable here


        # Instantiate model with numestimators decision trees
        rf = RandomForestClassifier(n_estimators=numestimators, random_state=0)
        # Train the model on training data
        rf.fit(X_train, y_train)

        # Use the forest's predict method on the test data
        predictions = rf.predict(X_test)

        print("-" * 40)
        print("test:")
        print(y_test)
        print("pred: ")
        print(predictions)
        print("-" * 40)
        accuracy = predictions.tolist().count(y_test.iloc[0])/len(predictions.tolist())
        print("accuracy: ", accuracy)
        # List of tuples with variable and importance
        importances = list(rf.feature_importances_)
        important = pd.DataFrame()
        important['value'] = feature_list
        important['importances'] = importances
        important['id'] = str(ids)
        #
        # return errors, RMSE, MAPerror, accuracy, important
        return accuracy, important

    def LOOCV_RF(self, dropcols=[], numestimators=5, fs=0.02):
        data = self.data
        idcolumn = "id_based_on_tree_name"
        outcomevar = self.outcomevar

        IDlist = list(data[idcolumn])
        drop = [idcolumn]  # add idcolumn to dropcols to drop from model
        drop = drop + dropcols

        acc = []
        importances = pd.DataFrame(columns=['value', 'importances', 'id'])

        # Run LOOCV Random Forest!
        for i in IDlist:
            accuracy, imp = self.RFLOOCV(data, i, outcomevar, drop, idcolumn, numestimators, fs)
            acc.append(accuracy)
            importances = importances.append(imp)
            idt = str(i)
            print('...' + idt + ' processing complete.')

        meanaccuracy = np.mean(acc)
        print('Mean Accuracy:' + str(meanaccuracy))
        return meanaccuracy, importances

class RandomForestOnly:
    def __init__(self, train_data, test_data, name):
        self.name = name
        self.train_data = train_data
        self.X_train, self.y_train = train_data.iloc[:,2:-1], train_data.iloc[:,1]
        self.test_data = test_data
        self.X_test, self.y_test = test_data.iloc[:,2:-1], test_data.iloc[:,1]
        self.num_estimators = NUMESTIMATORS
        self.features_to_use: list = None
        self.rf_model = None
        self.y_pred = None

    def get_feature_importances(self):
        self.features_to_use = pd.DataFrame(self.rf_model.feature_importances_,
                                            index = self.X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
        self.features_to_use.to_csv(f"importances/{self.name}_feature_importance.csv")

    def train_rf_model(self):
        self.rf_model = RandomForestClassifier(n_estimators = self.num_estimators, criterion = 'gini', random_state = 21, max_depth = 6)
        self.rf_model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.rf_model.predict(self.X_test)


    def get_results(self):
        contigency_matrix = pd.crosstab(self.y_test, self.y_pred, rownames=['Actual Species'], colnames=['Predicted Species'])
        # print(contigency_matrix)
        fig, ax = plt.subplots(figsize=(10, 10))
        sb.heatmap(ax=ax, data=contigency_matrix, annot=True, cmap="Reds")
        plt.tight_layout()
        plt.title(f'{self.name} trained on Dec20, tested on May21')
        plt.text(-3.4,25.5, f'{str(self.train_data.columns)}')
        acc_score = accuracy_score(self.y_test, self.y_pred)
        bal_acc_score = balanced_accuracy_score(self.y_test, self.y_pred)
        plt.text(8.4,24.5, f'acc score: {str(round(acc_score,2))}')
        plt.text(8.4,25.5, f'balanced acc score: {str(round(bal_acc_score,2))}')
        plt.savefig("rf_results/trained_dec_tested_may_glcm.png")
        plt.show()
        print("accuracy: ", accuracy_score(self.y_test, self.y_pred))
        print("balanced_accuracy: ", balanced_accuracy_score(self.y_test, self.y_pred))
        # report = pd.DataFrame(classification_report(self.y_test, self.y_pred, zero_division=1, output_dict=True)).transpose()
        # print(report)
        # plt.savefig('mytable.png')
        return acc_score, bal_acc_score

    def execute(self):
        self.train_rf_model()
        self.predict()
        acc_score, bal_acc_score = self.get_results()
        # self.get_feature_importances()
        return acc_score, bal_acc_score


class RandomForestLOOCV:
    def __init__(self, data):
        self.data = self.encode_tree_species(data)
        self.accuracy = None # percentage of it getting it right.
        self.results: pd.DataFrame = None

    def encode_tree_species(self, data):
        '''
        uses LabelEncoder from sklearn.preprocessing to encode the species. Note that duplicated species names have the same ID
        :param data:
        :return:
        '''
        species = data["tree_name"]
        self.le = LabelEncoder()
        self.le.fit(species)
        encoded_species = self.le.transform(species)

        data_new = data.copy()
        data_new["id_based_on_tree_name"] = encoded_species
        return data_new

    def execute_LOOCV(self):
        """
        Execute random forest for each sample as a test dataset, and all else a train dataset.
        :return: accuracy
        """
        test = []
        pred = []
        true_count = 0
        print(self.data.shape[0])
        for i, row in self.data.iterrows():

            test_data = self.data.iloc[[i],:]

            train_data = self.data.drop(self.data.index[i])

            X_test = test_data.drop(columns=["tree_name", "id_based_on_tree_name"])
            y_test = test_data["id_based_on_tree_name"]
            X_train = train_data.drop(columns=["tree_name", "id_based_on_tree_name"])
            y_train = train_data["id_based_on_tree_name"]

            rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=21,
                                                   max_depth=6)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            #
            # if (y_pred == y_test):
            #     true_count += 1
            y_t = self.le.inverse_transform(y_test)
            y_p = self.le.inverse_transform(y_pred)

            if (y_t == y_p):
                true_count += 1
            test.append(y_t)
            pred.append(y_p)
        print(true_count)
        self.accuracy = true_count/ self.data.shape[0]

        print(f"the accuracy is: {self.accuracy}")
        self.results = pd.DataFrame(list(zip(test, pred)), columns=["test", "pred"])
        return self.results



