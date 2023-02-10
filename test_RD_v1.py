import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


traindata_filepath = "./bank-additional-full.csv"
df = pd.read_csv(traindata_filepath, sep=";")

################### Data Analisis ############################

print(df.describe())
print(df.head())
print(df.columns)
print(f"columns length: {len(df.columns)}")
print(df.dtypes)

categorical_col = []
non_use_col = []

for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) > 1:
        categorical_col.append(column)
    elif len(df[column].unique()) == 1:
        non_use_col.append(column)

int_col = df.select_dtypes(include="int").columns.values
float_col = df.select_dtypes(include="float").columns.values

print(f"Int columns: {len(int_col)} ; {int_col}")
print(f"Float columns: {len(float_col)} ; {float_col}")
print(f"Categorical columns: {len(categorical_col)} ; {categorical_col}")
print(f"Non-useful columns {len(non_use_col)} ; {non_use_col}")


# Duration: Important note: this attribute highly affects the output target, so we drop it and use it as a final check

non_use_col = "duration"
int_col = np.delete(int_col, 1)
categorical_col= np.delete(categorical_col, -1) #we take of the list the target column "y"
print(f"Int columns without \"duration\": {len(int_col)} ; {int_col}")


####################### # Missing Data?################################

fig = plt.figure()
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap="viridis").set(title="All Missing Data")

##################### Observing Data ######################################

fig1 = plt.figure(figsize=(20, 20))
for i in range(0, len(float_col)):
    ax = fig1.add_subplot(5, 2, i + 1)
    sns.countplot(x=float_col[i], hue="y", data= df)
    ax.set_xlabel(float_col[i])

fig2 = plt.figure(figsize=(20, 20))
for i in range(0, len(int_col)):
    ax = fig2.add_subplot(5, 2, i + 1)
    sns.countplot(x=int_col[i], hue="y", data= df)
    ax.set_xlabel(int_col[i])

fig3 = plt.figure(figsize=(20, 20))
for i in range(0, len(categorical_col)):
    ax = fig3.add_subplot(5, 2, i + 1)
    sns.countplot(x=categorical_col[i], hue="y", data= df)
    plt.xticks(rotation=45)
    ax.set_xlabel(categorical_col[i])

############################## Preprossesor #######################################

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), float_col),
        ("int", StandardScaler(), int_col),
        ("cat", OneHotEncoder(drop="first"), categorical_col)
    ]
)

########################### # Append classifier to preprocessing pipeline.##############################

from sklearn.linear_model import LogisticRegression

clf_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=200)),
    ]
)

from sklearn.ensemble import RandomForestClassifier

clf_rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=120)),
    ]
)

from sklearn.neighbors import KNeighborsClassifier

clf_kn = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=10)),
    ]
)


################################## Train Test split ############################################

X = df.drop(["y"], axis=1)
y = df["y"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

dur_train = X_train["duration"]
dur_test = X_test["duration"]

X_train = X_train.drop(["duration"], axis=1)
X_test = X_test.drop(["duration"], axis=1)


################################### Predictions LR #####################################################
clf_lr.fit(X_train, y_train)
predictions_lr = clf_lr.predict(X_test)

################################### Predictions RF ##############################################
clf_rf.fit(X_train, y_train)
predictions_rf = clf_rf.predict(X_test)

################################### Predictions KN ######################################################
clf_kn.fit(X_train, y_train)
predictions_kn = clf_kn.predict(X_test)


################################## Classification Report #################################################
from sklearn.metrics import classification_report

clf_report_lr = classification_report(y_test, predictions_lr, output_dict=True)
clf_report_rf = classification_report(y_test, predictions_rf, output_dict=True)
clf_report_kn = classification_report(y_test, predictions_kn, output_dict=True)

classrep_lr = plt.figure()
plt.title("Class Report LR")
sns.heatmap(pd.DataFrame(clf_report_lr).iloc[:-1, :].T, annot=True)
plt.savefig("./class_rep_lr.jpg")

classrep_rf = plt.figure()
plt.title("Class Report RF")
sns.heatmap(pd.DataFrame(clf_report_rf).iloc[:-1, :].T, annot=True)
plt.savefig("./class_rep_rf.jpg")

classrep_kn = plt.figure()
plt.title("Class Report KN")
sns.heatmap(pd.DataFrame(clf_report_kn).iloc[:-1, :].T, annot=True)
plt.savefig("./class_rep_kn.jpg")


########################################### Confussion Matrix #################################

from sklearn.metrics import ConfusionMatrixDisplay

figlr= plt.figure()
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions_lr,
    cmap=plt.cm.Blues,
    # normalize="true"
)
disp.ax_.set_title("ConfusionMatrixDisplay_LR")
plt.savefig("./conf_matrix_lr.jpg")


figrf = plt.figure()
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions_rf,
    cmap=plt.cm.Blues,
    # normalize="true"
)
disp.ax_.set_title("ConfusionMatrixDisplay_RF")
plt.savefig("./conf_matrix_rf.jpg")


figkn = plt.figure()
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions_kn,
    cmap=plt.cm.Blues,
    # normalize="true"
)
disp.ax_.set_title("ConfusionMatrixDisplay_KN")
plt.savefig("./conf_matrix_kn.jpg")

############################################################################################################
#we chouse RandomForest because is more precise in respcet to de predicted value "yes", so one can focus the marketing campaing in thouse posible clients

clf_rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200)),
    ]
)
clf_rf.fit(X_train, y_train)
predictions_rf_op = clf_rf.predict(X_test)

classrep_rf_op = plt.figure()
plt.title("Class Report RF")
sns.heatmap(pd.DataFrame(clf_report_rf).iloc[:-1, :].T, annot=True)
plt.savefig("./class_rep_rf_op.jpg")

figrf_op = plt.figure()
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions_rf_op,
    cmap=plt.cm.Blues,
    # normalize="true"
)
disp.ax_.set_title("ConfusionMatrixDisplay_RF_op")
plt.savefig("./conf_matrix_rf_op.jpg")


################################### Compare results with "duration" column for RF ######################################
fig1rf= plt.figure(figsize=(20,10))
sns.scatterplot(x=dur_test, hue=predictions_rf, y=y_test).set(title="Scatterplot_Duration_ytest_predictions_RF")
plt.savefig("./Scatterplot_duration_ytest_predictions_RF.jpg")

fig2rf = plt.figure(figsize=(20, 10))
sns.displot(x=dur_test, y=predictions_rf, hue=y_test).set(title="Displot_duration_predictions_ytest_RF")
plt.savefig("./Displot_duration_predictions_ytest_RF.jpg")


plt.show()
