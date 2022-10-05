#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo : 08 - Opiniones y clasificación de textos
#
# Módulos necesarios:
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   SCIKIT-LEARN : 0.21.0
#   NLTK : 3.4
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------

#Carga del archivo
import pandas as pnd
mensajesTwitter = pnd.read_csv("datas/calentamientoClimatico.csv", delimiter=";")

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

#Información sobre la cantidad de observaciones y su contenido
print(mensajesTwitter.shape)
print(mensajesTwitter.head(2))

#Transformación de la característica Creencia
mensajesTwitter['CREENCIA'] = (mensajesTwitter['CREENCIA']=='Yes').astype(int)
print(mensajesTwitter.head(100))

#Función de normalización
import re
def normalizacion(mensaje):
    mensaje = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', mensaje)
    mensaje = re.sub('@[^\s]+','USER', mensaje)
    mensaje = mensaje.lower().replace("ё", "е")
    mensaje = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', mensaje)
    mensaje = re.sub(' +',' ', mensaje)
    return mensaje.strip()


#Normalización
mensajesTwitter["TWEET"] = mensajesTwitter["TWEET"].apply(normalizacion)
print(mensajesTwitter.head(10))

#Carga de StopWords
from nltk.corpus import stopwords
stopWords = stopwords.words('english')

#Eliminación de las Stops Words en las distintas frases
mensajesTwitter['TWEET'] = mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([palabra for palabra in mensaje.split() if palabra not in (stopWords)]))
print(mensajesTwitter.head(10))


#Aplicación de stemming
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
mensajesTwitter['TWEET'] = mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([stemmer.stem(palabra) for palabra in mensaje.split(' ')]))
print(mensajesTwitter.head(10))


#Lematización
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
mensajesTwitter['TWEET'] = mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([lemmatizer.lemmatize(palabra) for palabra in mensaje.split(' ')]))
print(mensajesTwitter.head(10))

print("¡Fin de la preparación!")


#Conjunto de aprendizaje y de prueba:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mensajesTwitter['TWEET'].values,  mensajesTwitter['CREENCIA'].values,test_size=0.2)


#Creación de la canalización de aprendizaje
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('algoritmo', MultinomialNB())])


#Aprendizaje
modelo = etapas_aprendizaje.fit(X_train,y_train)

from sklearn.metrics import classification_report
print(classification_report(y_test, modelo.predict(X_test), digits=4))

#Frase nueva:
frase = "Why should trust scientists with global warming if they didnt know Pluto wasnt a planet"
print(frase)

#Normalización
frase = normalizacion(frase)

#Eliminación de las stops words
frase = ' '.join([palabra for palabra in frase.split() if palabra not in (stopWords)])

#Aplicación de stemming
frase =  ' '.join([stemmer.stem(palabra) for palabra in frase.split(' ')])

#Lematización
frase = ' '.join([lemmatizer.lemmatize(palabra) for palabra in frase.split(' ')])
print (frase)

prediccion = modelo.predict([frase])
print(prediccion)
if(prediccion[0]==0):
    print(">> No cree en el calentamiento climático...")
else:
    print(">> Cree en el calentamiento climático...")



#------ Uso de SVM ---

#Definición de la canalización
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('algoritmo', svm.SVC(kernel='linear', C=2))])


#Aprendizaje
modelo = etapas_aprendizaje.fit(X_train,y_train)
from sklearn.metrics import classification_report
print(classification_report(y_test, modelo.predict(X_test), digits=4))

#Búsqueda del mejor parámetro C
from sklearn.model_selection import GridSearchCV
parametrosC = {'algoritmo__C':(1,2,4,5,6,7,8,9,10,11,12)}

busquedaCOptimo = GridSearchCV(etapas_aprendizaje, parametrosC,cv=2)
busquedaCOptimo.fit(X_train,y_train)
print(busquedaCOptimo.best_params_)


#Parámetro nuevo C=1
etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('algoritmo', svm.SVC(kernel='linear', C=1))])

modelo = etapas_aprendizaje.fit(X_train,y_train)
from sklearn.metrics import classification_report
print(classification_report(y_test, modelo.predict(X_test), digits=4))