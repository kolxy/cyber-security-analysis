import pandas as pd
import util
from constants import DataDir
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PATH_OUTPUT = os.getcwd() + "/output/"

def main():
    print("Reading data")
    training = util.get_clean_dataframe_from_file(DataDir.all_tables)
    training = util.convert_input_column_type(training)

    # drop them for good
    training = training.drop(["srcip", "sport", "dstip", "dsport"], axis = 1)

    # use training data for prediction
    x_train, y_train = util.get_input_output(training, class_type='binary')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
    top_features = run_logistic_regression(x_train, y_train, x_test, y_test)

    x_train, y_train = util.get_input_output(training, class_type='binary')
    x_train = x_train[[x[0] for x in top_features]]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)
    run_logistic_regression(x_train, y_train, x_test, y_test, True)

def run_logistic_regression(x_train, y_train, x_test, y_test, reduced = False):
    clf = LogisticRegression()
    print("Fitting")
    clf.fit(x_train, y_train)
    print("Predicting")
    score = clf.score(x_test, y_test)
    print(f"Accuracy: {score}")

    if not reduced:
        print("Generating graph")

        # create importance dictionary as {name: importance}
        feature_dict = dict(zip(x_train.columns, np.std(x_train, 0) * clf.coef_[0]))
        feature_importance = [(k, v) for k, v in feature_dict.items()]
        sorted_list = sorted(feature_importance, key = lambda x: abs(x[1]), reverse = True)

        top = sorted_list[:10]
        top_names = [x[0] for x in top]
        top_values = [x[1] for x in top]

        # output graph
        plt.barh(top_names, top_values)
        plt.gca().invert_yaxis()
        plt.title("Logistic regression feature importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature name")
        for index, value in enumerate(top_values):
            plt.text(value, index, str("{:.2e}".format(float(value))))
        plt.savefig(PATH_OUTPUT + "Logistic Regression feature importance.png")
        print("Output: ")
        print(PATH_OUTPUT + "Logistic Regression feature importance.png")
        
        return top[:1] # This shit is so powerful, one feature is enough to predict

if __name__ == '__main__':
    main()
