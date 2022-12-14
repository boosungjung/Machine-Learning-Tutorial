import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

def getBest(data):
    global y_test, y_train, x_test, x_train
    x_columns = ["G1", "G2", "G3", "studytime", "absences", "failures"]
    data = data[x_columns]

    predict = "G3"  # we are trying to predict G3
    X = np.array(data.drop([predict], axis=1))  # create an array without predict
    Y = np.array(data[predict])

    best = 0
    for _ in range(100):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)  # get the linear regression model
        acc = linear.score(x_test, y_test)
        if acc > best:
            best = acc
            with open("studentmodel.pickle", "wb") as f:
                pickle.dump(linear, f)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    # print("Coefficient: ", linear.coef_)
    # print("Intercept: ", linear.intercept_)
    data = pd.read_csv("./student/student-mat.csv", sep=";")

    x_train, x_test, y_train, y_test = getBest(data)

    pickle_in = open("studentmodel.pickle", "rb")
    linear = pickle.load(pickle_in)
    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(predictions[x],x_test[x],y_test[x])

    p = "G1"
    style.use("ggplot")
    pyplot.scatter(data[p],data["G3"])
    pyplot.xlabel(p)
    pyplot.ylabel("Final Grade")
    pyplot.show()

