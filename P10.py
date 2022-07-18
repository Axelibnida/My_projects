#!/usr/bin/env python
# coding: utf-8

# # Détection de faux billets avec Python
# ### 
# <figure>
#    <img src="./images/ONCFM logo.png" >
# </figure>
# 
# ## [Cahier des charges](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-analyst/DAN-P10-cdc-detection-faux-billets.pdf)
# ## Sommaire
#  - [1. Préparation](#1)
#    - [1.1 Importation des librairies](#1.1)
#    - [1.2 Importation des bases de données](#1.2)
#    - [1.3 Aperçu des données](#1.3)
#    - [1.4 Recherche de potentiels "outliers" dans les données](#1.4)
#  - [2. Régression linéaire](#2)
#    - [2.1 Préparation du modèle](#2.1)
#    - [2.2 Prédiction des valeurs manquantes](#2.2)
#    - [2.3 DataFrame complète avec les données manquantes](#2.3)
#  - [3. Régression logistique](#3)
#    - [3.1 Préparation du modèle](#3.1)
#    - [3.2 Matrice de confusion](#3.2)
#    - [3.3 Prédictions sur le test dataset](#3.3) 
#  - [4. K-Means clustering for classification](#4)
#    - [4.1 Préparation du modèle](4.1)
#    - [4.2 Matrice de confusion](#4.2)
#    - [4.3 Prédictions sur le test dataset](#4.3)
#  - [5. Application finale](#5)
# 
# <a name="1"></a>
# ## 1. Préparation
# <a name="1.1"></a>
# ### 1.1 Importation des librairies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# <a name="1.2"></a>
# ### 1.2 Importation des bases de données

# In[2]:


data = pd.read_csv('C:\\Users\\axeli\\OneDrive\\Desktop\\Projet 10\\billets.csv', sep=';')
data_test = pd.read_csv('C:\\Users\\axeli\\OneDrive\\Desktop\\Projet 10\\billets_production.csv', sep=',')

test3 = pd.read_csv('C:\\Users\\axeli\\OneDrive\\Desktop\\Projet 10\\billets_test.csv', sep=',')


# Nous séparons les faux billets des vrais billets en deux bases de données :

# In[3]:


data_true = data[data['is_genuine'] == True]
data_false = data[data['is_genuine'] == False]


# <a name="1.3"></a>
# ### 1.3 Aperçu des données

# Dimensions de nos deux bases de données :

# In[4]:


print('Vrais billets :', data_true.shape)
print('Faux billets :', data_false.shape)


# Aperçu de la DataFrame des vrais billets :

# In[5]:


display(data_true)
print(data_true.info())


# Aperçu de la DataFrame des faux billets :

# In[6]:


display(data_false)
print(data_false.info())


# Nous pouvons aussi afficher une matrice de corrélation pour voir si certaines variables sont corrélées. Cela n'a pas l'air d'être le cas.

# In[7]:


corrMatrix = data.corr()
display(corrMatrix)


# <a name="1.4"></a>
# ### 1.4 Recherche de potentiels "outliers" dans les données

# In[8]:


def Outliers(i):
    """ Display un boxplot et la liste des outliers de la variable entrée """
    print()
    print("Boxplot :", i)
    boite = boxplot(data[i])
    plt.show()
    print()
    print("Liste des outliers :")
    top_points = boite["fliers"][0].get_data()[1]
    display(pd.DataFrame(data[i][data[i].isin(top_points)]).sort_values(by = i, ascending=False))


# In[9]:


for column in data.columns[1:]:
   Outliers(column)


# Création de DataFrames séparant les données complètes des données ayant 'margin_low' == null

# In[10]:


data_true_null = data_true[data_true["margin_low"].isnull()]
data_false_null = data_false[data_false["margin_low"].isnull()]

data_true_without_null = data_true[data_true["margin_low"].notnull()]
data_false_without_null = data_false[data_false["margin_low"].notnull()]


# <a name="2"></a>
# ## 2. Régression linéaire
# Nous allons prédire les valeurs manquantes de notre base de données grâce à une régression linéaire.
# <a name="2.1"></a>
# ### 2.1 Préparation du modèle
# Visualisation des données

# In[11]:


sns.pairplot(data_true_without_null, x_vars=["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"], y_vars="margin_low", kind='reg')
sns.pairplot(data_false_without_null, x_vars=["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"], y_vars="margin_low", kind='reg')


# Attribution de X et y

# In[12]:


feature_cols = ['diagonal', 'height_left', 'height_right', 'margin_up', 'length']

X_true = data_true_without_null[feature_cols]
X_false = data_false_without_null[feature_cols]

y_true = data_true_without_null['margin_low']
y_false = data_false_without_null['margin_low']


# Séparation des données en données de training et données de test

# In[13]:


X_true_train, X_true_test, y_true_train, y_true_test = train_test_split(X_true, y_true)
X_false_train, X_false_test, y_false_train, y_false_test = train_test_split(X_false, y_false)


# Apprentissage du modèle

# In[14]:


linreg_true = LinearRegression()
linreg_false = LinearRegression()

linreg_true.fit(X_true_train, y_true_train)
linreg_false.fit(X_false_train, y_false_train)


# Coefficients :

# In[15]:


print(linreg_true.intercept_)
print(linreg_false.intercept_)

print(linreg_true.coef_)
print(linreg_false.coef_)

zip(feature_cols, linreg_true.coef_)
zip(feature_cols, linreg_false.coef_)


# Prédiction du modèle entraîné sur les données de test :

# In[16]:


y_true_pred = linreg_true.predict(X_true_test)
y_false_pred = linreg_false.predict(X_false_test)

print(y_true_pred)
print(y_false_pred)


# Évaluation du modèle

# In[17]:


print()
print('Vrais billets :')
print(metrics.mean_absolute_error(y_true_test, y_true_pred))
print(metrics.mean_squared_error(y_true_test, y_true_pred))
print(np.sqrt(metrics.mean_squared_error(y_true_test, y_true_pred)))

print()
print('Faux billets :')
print(metrics.mean_absolute_error(y_false_test, y_false_pred))
print(metrics.mean_squared_error(y_false_test, y_false_pred))
print(np.sqrt(metrics.mean_squared_error(y_false_test, y_false_pred)))


# <a name="2.2"></a>
# ### 2.2 Prédiction des valeurs manquantes

# In[18]:


print()
print('Vrais billets :')

data_true_pred = data_true_null[feature_cols]
pred_true = linreg_true.predict(data_true_pred)
print(pred_true)

data_true2 = data_true.copy(deep=True)
data_true2.loc[data_true2['margin_low'].isnull(), 'margin_low'] = pred_true


print()
print('Faux billets :')

data_false_pred = data_false_null[feature_cols]
pred_false = linreg_false.predict(data_false_pred)
print(pred_false)

data_false2 = data_false.copy(deep=True)
data_false2.loc[data_false2['margin_low'].isnull(), 'margin_low'] = pred_false


# <a name="2.3"></a>
# ### 2.3 DataFrame complète avec les données manquantes

# In[19]:


frames = (data_true2, data_false2)
data2 = pd.concat(frames)

display(data2)
print(data2.info())


# <a name="3"></a>
# ## 3. Régression logistique
# <a name="3.1"></a>
# ### 3.1 Préparation du modèle
# Attribution de X et y

# In[20]:


feature_cols_log = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
X_log = data2[feature_cols_log]

data2['is_genuine'] = data2['is_genuine'].astype('category')
y_log = data2['is_genuine'].cat.codes


# Séparation des données en données de training et données de test

# In[21]:


X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.3, random_state=0)


# Apprentissage du modèle

# In[22]:


logreg = LogisticRegression()
logreg.fit(X_train_log, y_train_log)


# Prédiction du modèle entraîné sur les données de test

# In[23]:


y_pred_log = logreg.predict(X_test_log)
print(y_pred_log)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_log, y_test_log)))


# <a name="3.2"></a>
# ### 3.2 Matrice de confusion

# In[24]:


matrice_de_confusion_log = confusion_matrix(y_test_log, y_pred_log)
display(pd.DataFrame(matrice_de_confusion_log))

print(classification_report(y_test_log, y_pred_log))


# <a name="3.3"></a>
# ### 3.3 Prédictions sur le test dataset

# In[25]:


display(data_test)


# In[26]:


data_test2 = data_test[["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]]

pred_log = logreg.predict(data_test2)

print(pred_log) # 0 == faux billet, 1 == vrai billet


# <a name="4"></a>
# ## 4. K-MEANS CLUSTERING FOR CLASSIFICATION
# <a name="4.1"></a>
# ### 4.1 Préparation du modèle
# Attribution de X et y

# In[27]:


feature_cols_km = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
X_km = data2[feature_cols_km]

y_km = data2['is_genuine'].cat.codes


# Séparation des données en données d'entraînement et données de test

# In[28]:


X_train_km, X_test_km, y_train_km, y_test_km = train_test_split(X_km, y_km, test_size=0.3, random_state=0)


# Entraînement du modèle

# In[29]:


kmeans = KMeans(n_clusters=2, random_state =42)
kmeans.fit(X_train_km, y_train_km)


# Prédiction du modèle sur les données de test

# In[30]:


y_pred_km = kmeans.predict(X_test_km)
print(y_pred_km)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_km, y_test_km)))


# <a name="4.2"></a>
# ### 4.2 Matrice de confusion

# In[31]:


matrice_de_confusion_km = confusion_matrix(y_test_km, y_pred_km)
display(pd.DataFrame(matrice_de_confusion_log))

print(classification_report(y_test_km, y_pred_km))


# <a name="4.3"></a>
# ### 4.3 Prédictions sur le test dataset

# In[32]:


pred_km = kmeans.predict(data_test2)

print(pred_km) # 0 == vrai billet, 1 == faux billet


# Centroïdes des clusters :

# In[33]:


print(kmeans.cluster_centers_)


# <a name="5"></a>
# ## 5. Application finale

# In[34]:



def logistic_regression(matrix):
    """Réalise une régression logistique sur une matrice entrée"""
    print("Vérification des billets par Régression Logistique...")
    print()
    matrix2 = matrix[feature_cols_log]
    prediction = logreg.predict(matrix2)
    matrix['Prédiction'] = prediction
    matrix.loc[matrix.Prédiction == 0, "Prédiction"] = "Faux billet"
    matrix.loc[matrix.Prédiction == 1, "Prédiction"] = "Vrai billet"
    return matrix


def k_means_classification(matrix):
    """Réalise une classification via K-Means d'une matrice entrée"""
    print("Vérification des billets par Classification via K-Means...")
    print()
    matrix2 = matrix[feature_cols_km]
    prediction = kmeans.predict(matrix2)
    matrix['Prédiction'] = prediction
    matrix.loc[matrix.Prédiction == 0, "Prédiction"] = "Vrai billet"
    matrix.loc[matrix.Prédiction == 1, "Prédiction"] = "Faux billet"
    return matrix


# In[35]:


data_test_reg = data_test.copy(deep=True)
data_test_km = data_test.copy(deep=True)


# In[36]:


logistic_regression(test3)


# In[37]:


k_means_classification(test3)


# In[ ]:




