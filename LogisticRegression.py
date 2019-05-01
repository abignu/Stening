
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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
#Logistic Regression
model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

#entrenamos
model.fit(train_data_X, train_data_Y)

#obtenemos score con el testing
score = model.score(test_data_X, test_data_Y)

print('porcentaje de acierto = ', score) #lo imprimimos

#hacemos predicciones y sacamos confusion matriz
predicciones = model.predict(test_data_X) #predecimos y comparamos con su target

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(test_data_Y, predicciones)

#plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title('ST: 0 , TM: 1, SY: 2; score = {}'.format(score))
#plt.title(all_sample_title, size = 15);
plt.show()