import os
import webbrowser

import pandas as pd
import pandas_profiling as pp
from flask import Flask
from flask import render_template, request

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024


def Regression():
    # Linear Regression
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    df = pd.read_csv(path())
    df = df.fillna(df.mean())
    # print(df)
    ind_var1 = int(input("1enter the column number where you want to start in independent variable"))
    ind_var2 = int(input("2enter the column number where you want to end in independent variable"))
    dep_var = int(input("2enter the column number where you want the dependent variable"))
    X = df.iloc[:, ind_var1:ind_var2]
    y = df.iloc[:, dep_var]
    # print(X)
    print(df)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    print("Training Size is ", X_train.shape)
    print("Testing Size is ", X_test.shape)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    # Summary of the predictions made by the regressor
    LinearReg_r2score = r2_score(y_test, y_pred)

    # Decision Tree Regression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    # df = pd.read_csv(path())
    # print(df)
    # ind_var1 = int(input("enter the column number where you want to start in independent variable"))
    # ind_var2 = int(input("enter the column number where you want to end in independent variable"))
    # dep_var = int(input("enter the column number where you want the dependent variable"))
    # X = df.iloc[:, list(range(ind_var1, ind_var2))]
    # y = df.iloc[:, dep_var]
    # print(X)
    # print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    # print("Training Size is ", X_train.shape)
    # print("Testing Size is ", X_test.shape)
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    # Summary of the predictions made by the Regressor
    DecisionTree_r2score = r2_score(y_test, y_pred)

    # Random Forest Regression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    # df = pd.read_csv(path())
    # print(df)
    # ind_var1 = int(input("enter the column number where you want to start in independent variable"))
    # ind_var2 = int(input("enter the column number where you want to end in independent variable"))
    # dep_var = int(input("enter the column number where you want the dependent variable"))
    # X = df.iloc[:, list(range(ind_var1, ind_var2))]
    # y = df.iloc[:, dep_var]
    # print(X)
    # print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    # print("Training Size is ", X_train.shape)
    # print("Testing Size is ", X_test.shape)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    # Summary of the predictions made by the Regressor
    RandomForest_r2score = r2_score(y_test, y_pred)

    print("Linear Regression R2 score is ", LinearReg_r2score * 100, "%")
    print("Decision Tree Regression R2 score is ", DecisionTree_r2score * 100, "%")
    print("Random Forest Regression R2 score is ", RandomForest_r2score * 100, "%")


def Classification():
    # Logistic Regression
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    df = pd.read_csv(path())
    df = df.fillna(df.mean())
    print(df)
    ind_var1 = int(input("0enter the column number where you want to start in independent variable"))
    ind_var2 = int(input("4enter the column number where you want to end in independent variable"))
    dep_var = int(input("5enter the column number where you want the dependent variable"))
    X = df.iloc[:, list(range(ind_var1, ind_var2))]
    y = df.iloc[:, dep_var]
    # print(X)
    # print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    print("Training Size is ", X_train.shape)
    print("Testing Size is ", X_test.shape)
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score
    Logitsic_reg_Acc = 'Logistic Regression Accuracy Score is {:.4%}'.format(accuracy_score(y_pred, y_test))
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    # plt.show()
    plt.savefig("LogisticRegressionGraph.png")

    # SVM Code
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    # df = pd.read_csv(path())
    # print(df)
    # ind_var1 = int(input("enter the column number where you want to start in independent variable"))
    # ind_var2 = int(input("enter the column number where you want to end in independent variable"))
    # dep_var = int(input("enter the column number where you want the dependent variable"))
    # X = df.iloc[:, list(range(ind_var1, ind_var2))]
    # y = df.iloc[:, dep_var]
    # print(X)
    # print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    # print("Training Size is ", X_train.shape)
    # print("Testing Size is ", X_test.shape)
    from sklearn.svm import SVC
    classifier = SVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score
    SVM_Acc = 'SVM Accuracy Score is {:.4%}'.format(accuracy_score(y_pred, y_test))
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    # plt.show()
    plt.savefig("SVMGraph.png")

    # KNN Code
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    # df = pd.read_csv(path())
    # print(df)
    # ind_var1 = int(input("enter the column number where you want to start in independent variable"))
    # ind_var2 = int(input("enter the column number where you want to end in independent variable"))
    # dep_var = int(input("enter the column number where you want the dependent variable"))
    # X = df.iloc[:, list(range(ind_var1, ind_var2))]
    # y = df.iloc[:, dep_var]
    # print(X)
    # print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    # print("Training Size is ", X_train.shape)
    # print("Testing Size is ", X_test.shape)
    # K-Nearest Neighbours
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score
    KNN_Acc = 'KNN Accuracy Score is {:.4%}'.format(accuracy_score(y_pred, y_test))
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    # plt.show()
    plt.savefig("KNNGraph.png")

    # Naive Bayes
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    # df = pd.read_csv(path())
    # print(df)
    # ind_var1 = int(input("enter the column number where you want to start in independent variable"))
    # ind_var2 = int(input("enter the column number where you want to end in independent variable"))
    # dep_var = int(input("enter the column number where you want the dependent variable"))
    # X = df.iloc[:, list(range(ind_var1, ind_var2))]
    # y = df.iloc[:, dep_var]
    # print(X)
    # print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    # print("Training Size is ", X_train.shape)
    # print("Testing Size is ", X_test.shape)
    # Gaussian Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score
    NB_Acc = 'Naive Bayes Accuracy Score is {:.4%}'.format(accuracy_score(y_pred, y_test))
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    # plt.show()
    plt.savefig("GaussianNaiveBayesGraph.png")

    # Decision Tree Code
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    # df = pd.read_csv(path())
    # print(df)
    # ind_var1 = int(input("enter the column number where you want to start in independent variable"))
    # ind_var2 = int(input("enter the column number where you want to end in independent variable"))
    # dep_var = int(input("enter the column number where you want the dependent variable"))
    # X = df.iloc[:, list(range(ind_var1, ind_var2))]
    # y = df.iloc[:, dep_var]
    # print(X)
    # print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    # print("Training Size is ", X_train.shape)
    # print("Testing Size is ", X_test.shape)
    # Decision Tree's
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score
    DecisionTree_Acc = 'Decision Tree Accuracy Score is {:.4%}'.format(accuracy_score(y_pred, y_test))
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    # plt.show()
    plt.savefig("DecisionTreeGraph.png")

    # Random Forest Code
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    # df = pd.read_csv(path())
    # print(df)
    # ind_var1 = int(input("enter the column number where you want to start in independent variable"))
    # ind_var2 = int(input("enter the column number where you want to end in independent variable"))
    # dep_var = int(input("enter the column number where you want the dependent variable"))
    # X = df.iloc[:, list(range(ind_var1, ind_var2))]
    # y = df.iloc[:, dep_var]
    # print(X)
    # print(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
    # print("Training Size is ", X_train.shape)
    # print("Testing Size is ", X_test.shape)
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(random_state=23)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score
    RandomForest_Acc = 'Random Forest Accuracy Score is {:.4%}'.format(accuracy_score(y_pred, y_test))
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    # plt.show()
    plt.savefig("RandomForestGraph.png")

    print(KNN_Acc)
    print(NB_Acc)
    print(SVM_Acc)
    print(Logitsic_reg_Acc)
    print(DecisionTree_Acc)
    print(RandomForest_Acc)


def Clustering():
    #Kmeans
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    import seaborn as sns
    df = pd.read_csv(path())
    df = df.fillna(df.mean())
    print(df)
    ind_var1 = int(input("0enter the column number where you want to start in independent variable"))
    ind_var2 = int(input("10enter the column number where you want to end in independent variable"))
    dep_var = int(input("0enter the column number where you want the dependent variable"))
    cluster= int(input("4enter the number of clusters "))
    X = df.iloc[:, list(range(ind_var1, ind_var2))]
    y = df.iloc[:, dep_var]
    #print(X)
    #print(y)
    le = LabelEncoder()
    X['status_type'] = le.fit_transform(X['status_type'])
    y = le.transform(y)
    cols = X.columns
    ms = MinMaxScaler()
    X = ms.fit_transform(X)
    kmeans = KMeans(n_clusters=cluster, random_state=0)
    km= kmeans.fit(X)
    print("",kmeans.cluster_centers_)
    print("Inertia is : ",kmeans.inertia_)
    labels = kmeans.labels_
    # check how many of the samples were correctly labeled
    correct_labels = sum(y == labels)
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    print('Accuracy score: {0:0.2f}%'.format(correct_labels*100 / float(y.size)))
    cs = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        cs.append(kmeans.inertia_)
    plt.plot(range(1, 11), cs)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('CS')
    plt.savefig('Kmeans_Elbow.png')


def path():
    for file in request.files.getlist('files[]'):
        print(os.getcwd())
        file.save(os.path.join(os.getcwd(), 'input', file.filename))
    files = [x for x in
             os.listdir(r'C:/Users/Hamza/PycharmProjects/pythonProject1/Data Visualisation and Modelling/input')
             if x.endswith(".csv")]
    file_path = 'C:/Users/Hamza/PycharmProjects/pythonProject1/Data Visualisation and Modelling/input/' + str(files[0])
    return file_path


'''
@app.route("/")
def loadPage():
    return render_template('output.html') '''


@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return render_template('output.html', error=error)

    return render_template('login.html', error=error)


@app.route("/visualize", methods=['POST'])
def datavisual():
    df = pd.read_csv(path())
    report = pp.ProfileReport(df)
    report.to_file('analysis.html')
    webbrowser.open_new_tab('analysis.html')
    return render_template('train.html')


@app.route("/train", methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        choice = request.form.get('fname')
        # print(choice)
        # print(type(choice))
        # print(path())
        if choice == '1':
            Regression()
        elif choice == '2':
            Classification()
        elif choice == '3':
            Clustering()
        else:
            print('Please enter correct number for the respective model name')
    return render_template('train.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
