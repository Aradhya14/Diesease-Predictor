
# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# Reading the train.csv by removing the 
# last column since it's an empty column
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
	"Disease": disease_counts.index,
	"Counts": disease_counts.values
})
# to plot in graphical and beautiful format:

plt.figure(figsize = (18,8))# plot figure
sns.barplot(x = "Disease", y = "Counts", data = temp_df) # defining x axis annd y axis plots
plt.xticks(rotation=90) # to rotate labels of x axis to 90 degree to avoid overlapping used incases s labels are long name
plt.show()  # show the plot



#encoding the prognosis cuz it is the target column with object datatype which is not supported by ML hence label encoding is done t
# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])


X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
X, y, test_size = 0.2, random_state = 24)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")
print(y)



# Defining scoring metric for k-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Initializing Models
models = {
"SVC":SVC(),
"Gaussian NB":GaussianNB(),
"Random Forest":RandomForestClassifier(random_state=18)
}

# Producing cross validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv = 10, 
                             n_jobs = -1, 
                             scoring = cv_scoring)
    print("=="*30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")



#training of data by fitting nd checking by prediction on SVM model
svmModel = SVC()
svmModel.fit(X_train,y_train)
preds = svmModel.predict(X_test)
preds


confMatrix = confusion_matrix(y_test,preds)   # defining condusion matrix for test data here stored in ytest and predictions made by train on x data by model fitting by svm
plt.figure(figsize = (10,5))  # initiating the figure
sns.heatmap(confMatrix , annot = True)   # annot = true: provides the entire matrix accept the defined values with true as 0
plt.title("confusion maatrix on y test data")
plt.show()



# training and testing on bayes model using Gaussian Nb
nbmodel = GaussianNB()
nbmodel.fit(X_train,y_train)
pred = nbmodel.predict(X_test)

cf = confusion_matrix(y_test, pred)
plt.figure(figsize=(12,8))
sns.heatmap(cf, annot=True)
plt.show()


#training and testing on random forest model
rfmodel= RandomForestClassifier()
rfmodel.fit(X_train, y_train)
pred= rfmodel.predict(X_test)

conf= confusion_matrix(y_test, pred)
plt.figure(figsize=(12,8))
sns.heatmap(conf, annot= True)
plt.show()



# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading the test data
test_data = pd.read_csv("Training.csv").dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

# Making prediction by take mode of predictions 
# made by all the classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

final_preds = [mode([i,j,k]) for i,j,k in zip(svm_preds, nb_preds, rf_preds)]
final_preds



symptoms = X.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}



# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)

    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][finalrf.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][finalGNB.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][finalSVM.predict(input_data)[0]]

    # making final prediction by taking mode of all predictions
    #final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    # Calculate the mode using np.unique
    data = [rf_prediction, nb_prediction, svm_prediction]
    values, counts = np.unique(data, return_counts=True)
    mode_index = np.argmax(counts)
    mode = values[mode_index]
    final_prediction = mode
    #final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return predictions

# Testing the function
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))






