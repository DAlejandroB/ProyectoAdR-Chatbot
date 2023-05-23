import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import SnowballStemmer

# Cargar datos de entrenamiento
with open('intenciones.json') as archivo:
    datos = json.load(archivo)

# Preprocesamiento de texto
stemmer = SnowballStemmer('spanish')
intenciones = datos['intenciones']
X = []
y = []

for intencion in intenciones:
    etiqueta = intencion['etiqueta']
    patrones = intencion['patrones']
    for patron in patrones:
        X.append(str(patron))  # Asegurar que el patrón sea una cadena de texto
        y.append(etiqueta)

vectorizador = TfidfVectorizer(tokenizer=lambda texto: [stemmer.stem(palabra) for palabra in texto.split()])
X_train = vectorizador.fit_transform(X)  # Obtener vectores de entrenamiento
y_train = np.array(y)  # Obtener etiquetas de entrenamiento

# Entrenar modelo clasificador
clasificador = RandomForestClassifier()
clasificador.fit(X_train, y_train)

# Función para clasificar la intención del usuario
def clasificar_intencion(texto):
    X = vectorizador.transform([texto])
    similitudes = cosine_similarity(X, X_train)
    mejor_similitud = np.max(similitudes)
    mejor_coincidencia_idx = np.argmax(similitudes)
    mejor_etiqueta = y_train[mejor_coincidencia_idx]
    respuestas = [r['respuestas'] for r in datos['intenciones'] if r['etiqueta'] == mejor_etiqueta]
    return random.choice(respuestas[0])  # Devolver una respuesta aleatoria

# Función para interactuar con el chatbot
def interactuar():
    print("¡Hola! Soy un chatbot. Puedes comenzar a hacer preguntas o tener una conversación conmigo. Escribe 'salir' para terminar.")

    while True:
        entrada = input("Tú: ")

        if entrada.lower() == 'salir':
            print("Chatbot: Hasta luego. ¡Que tengas un buen día!")
            break

        # Responder según la mejor coincidencia encontrada
        respuesta = clasificar_intencion(entrada)
        print("Chatbot: " + respuesta)

# Iniciar la interacción con el chatbot
interactuar()
