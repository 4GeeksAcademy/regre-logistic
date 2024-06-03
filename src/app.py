from utils import db_connect
engine = db_connect()

# your code here
# problema y creación del data frame:

import pandas as pd
pd.set_option('display.max_columns', None)
     

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep = ";")
data.head()

# limpieza y exploración de datos:

data.shape
     
data.info()

# Variables categorícas;

import matplotlib.pyplot as plt
import seaborn as sns
     

fig, axis = plt.subplots(3, 3, figsize=(20, 10))

# Creamos los histogramas
sns.histplot(ax = axis[0,0], data = data, x = "age")
sns.histplot(ax = axis[0,1], data = data, x = "marital")
sns.histplot(ax = axis[0,2], data = data, x = "education")
sns.histplot(ax = axis[1,0], data = data, x = "default")
sns.histplot(ax = axis[1,1], data = data, x = "housing")
sns.histplot(ax = axis[1,2], data = data, x = "loan")
sns.histplot(ax = axis[2,0], data = data, x = "contact")
sns.histplot(ax = axis[2,1], data = data, x = "month")
sns.histplot(ax = axis[2,2], data = data, x = "y")


# Ajustamos el diseño:
plt.tight_layout()

# Mostramos:
plt.show()

# vamos a comparar las categoricas con la columna 'Y' que es la target de contratación:

fig, axis = plt.subplots(3, 3, figsize = (15, 7))

sns.countplot(ax = axis[0, 0], data = data, x = "age", hue = "y")
sns.countplot(ax = axis[0, 1], data = data, x = "marital", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[0, 2], data = data, x = "education", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[1, 0], data = data, x = "default", hue = "y")
sns.countplot(ax = axis[1, 1], data = data, x = "housing", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[1, 2], data = data, x = "loan", hue = "y").set(ylabel = None)
sns.countplot(ax = axis[2, 0], data = data, x = "day_of_week", hue = "y")
sns.countplot(ax = axis[2, 1], data = data, x = "poutcome", hue = "y").set(ylabel = None)

plt.tight_layout()
fig.delaxes(axis[1, 2])

plt.show()

# vemos la relación de gente con estudios y sus contrataciones:



# Histogramas de variables numéricas:

# vamos a verla edad que más contrata:

fig, axis = plt.subplots(figsize = (20, 10))

sns.countplot(data = data, x = "age", hue = "y")

plt.show()

# la edad de 28 a 35 son adecuados para la contratación del producto

# Es importante el nivel de estudios?:

fig, axis = plt.subplots(figsize = (20, 10))

sns.countplot(data = data, x = "education", hue = "y")

plt.show()

# Análisis Numérico-Categórico:

from sklearn.preprocessing import MinMaxScaler

data["job_n"] = pd.factorize(data["job"])[0]
data["marital_n"] = pd.factorize(data["marital"])[0]
data["education_n"] = pd.factorize(data["education"])[0]
data["default_n"] = pd.factorize(data["default"])[0]
data["housing_n"] = pd.factorize(data["housing"])[0]
data["loan_n"] = pd.factorize(data["loan"])[0]
data["contact_n"] = pd.factorize(data["contact"])[0]
data["month_n"] = pd.factorize(data["month"])[0]
data["day_of_week_n"] = pd.factorize(data["day_of_week"])[0]
data["poutcome_n"] = pd.factorize(data["poutcome"])[0]
data["y_n"] = pd.factorize(data["y"])[0]
num_variables = ["job_n", "marital_n", "education_n", "default_n", "housing_n", "loan_n", "contact_n", "month_n", "day_of_week_n", "poutcome_n",
                 "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y_n"]
     

fig, axes = plt.subplots(figsize=(20, 10))

sns.heatmap(data[["job_n", "marital_n", "education_n", "default_n", "housing_n", "loan_n", "contact_n", "month_n", "day_of_week_n", "poutcome_n",
                 "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y_n"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()    

# Variables Numéricas:

fig, axis = plt.subplots(4, 2, figsize = (10, 14), gridspec_kw = {"height_ratios": [6, 1, 6, 1]})

sns.histplot(ax = axis[0, 0], data = data, x = "duration")
sns.boxplot(ax = axis[1, 0], data = data, x = "duration")

sns.histplot(ax = axis[0, 1], data = data, x = "campaign")
sns.boxplot(ax = axis[1,1], data = data, x = "campaign")

sns.histplot(ax = axis[2, 0], data = data, x = "previous")
sns.boxplot(ax = axis[3, 0], data = data, x = "previous")

sns.histplot(ax = axis[2,1], data = data, x = "pdays")
sns.boxplot(ax = axis[3, 1], data = data, x = "pdays")


plt.tight_layout()

plt.show()

# borro las columnas del dataframe que me valen para la opti:

columns_to_drop = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
data = data.drop(columns=columns_to_drop)
   
data

# Datos Atípicos:

data.describe()

duration_stats = data["duration"].describe()
duration_stats

duration_iqr = duration_stats["75%"] - duration_stats["25%"]
upper_limit = duration_stats["75%"] + 1.5 * duration_iqr
lower_limit = duration_stats["25%"] - 1.5 * duration_iqr

print(f"Los datos para los límites superior e inferior de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, su rango intercuartílico es de: {round(duration_iqr, 2)}")

# Vamos a eliminar:

atipicos_para_eliminar = data[data["duration"] > 644].index
     

data = data.drop(atipicos_para_eliminar)
     

data["duration"].describe()


# vamos con Campaign:

campaign_stats = data["campaign"].describe()
campaign_stats

campaign_iqr = campaign_stats["25%"]
upper_limit = campaign_stats["75%"] + 1.5 * campaign_iqr
lower_limit = campaign_stats["25%"] - 1.5 * campaign_iqr
print(f"Los datos para los límites superior e inferior de outliers son {round(upper_limit, 2)} y {round(lower_limit, 2)}, su rango intercuartílico es de: {round(campaign_iqr, 2)}")
     
# Atípicos nuevos 2:

atipicos_para_eliminar_2 = data[data["campaign"] > 4.5].index
     

data = data.drop(atipicos_para_eliminar_2)
     

data["campaign"].describe()

data

# Insull:

data.isnull().sum().sort_values(ascending = False)

# Vamos a escalar valores:


from sklearn.preprocessing import MinMaxScaler

num_variables = ["job_n", "marital_n", "education_n", "default_n", "housing_n", "loan_n", "contact_n", "month_n", "day_of_week_n", "poutcome_n",
                 "age", "duration", "campaign", "pdays", "previous", "y_n"]

scaler = MinMaxScaler()
scal_features = scaler.fit_transform(data[num_variables])
data_scal = pd.DataFrame(scal_features, index = data.index, columns = num_variables)
data_scal.head()

# Vamos a seleccionar caracteristicas para el Train:

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = data_scal.drop("y_n", axis = 1)
y = data_scal["y_n"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

selection_model = SelectKBest(chi2, k = 5)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel["y_n"] = list(y_train)
X_test_sel["y_n"] = list(y_test)
X_train_sel.to_csv("X_train_sel.csv", index=False)
X_test_sel.to_csv("X_test_sel.csv", index=False)

# Regresión Logistica:

train_data = pd.read_csv("X_train_sel.csv")
test_data = pd.read_csv("X_train_sel.csv")

train_data.head()

X_train = train_data.drop(["y_n"], axis = 1)
y_train = train_data["y_n"]
X_test = test_data.drop(["y_n"], axis = 1)
y_test = test_data["y_n"]
     

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

# Vamos con la optimización del modelo:

# Modelo de Cuadrícula:

from sklearn.model_selection import GridSearchCV
# expongo los paramentros que pretendo ajustar:

hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}
# lanzo la cuadrícula:

grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 10)
grid

# ajustamos el modelo de Machine con hiperparametros:

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

# ajusto el modelo y predigo la preción del modelo de regre/logist:

model = LogisticRegression(C = 0.1, penalty = "l2", solver = "liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

accuracy_score(y_test, y_pred)

# Modelo de ajuste externo( lo he buscado fuera) deajuste de busqueda aleatoria y sus predicciones:

model_random_search = LogisticRegression(penalty="l2", C=1000, solver="newton-cg")
model_random_search.fit(X_train, y_train)
y_pred = model_random_search.predict(X_test)

random_search_accuracy = accuracy_score(y_test, y_pred)
random_search_accuracy
