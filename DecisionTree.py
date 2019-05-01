
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
#acomodamos data
import pandas as pd
import numpy as np

#dataset ppal
df = pd.read_csv('1500x6.csv')

df = shuffle(df) #shuffle

#armamos los inputs
print('armamos los inputs...')
df_real = pd.DataFrame()


df_real = df[['Largo','Diametro','Anclajes','Numero Ramas','Ancho pared']].copy()

df_real = df_real.values #lo paso a array

#armamos los targets
target = df[['Clases']].copy()

'''
target = target.values #lo paso a array
comprobacion_target = comprobacion_target.values

target = np.ravel(target)

#codificamos las fuentes
le = preprocessing.LabelEncoder()
le.fit(target)
target = le.transform(target)
'''

train_data_X, test_data_X, train_data_Y, test_data_Y = train_test_split(df_real, target, shuffle=True, test_size=0.2) #20% es test data

#Decision Tree
model = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)

#entrenamos
model.fit(train_data_X, train_data_Y)

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')

#obtenemos score con el testing
predicciones = model.predict(test_data_X)

#accuracy
score =  accuracy_score(test_data_Y,predicciones)*100

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(test_data_Y, predicciones)

#plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Reds');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title('ST: 0 , TM: 1, SY: 2; score = {}'.format(score))
#plt.title(all_sample_title, size = 15);
plt.show()