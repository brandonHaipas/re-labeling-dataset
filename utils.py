
import openai
from openai import OpenAI
import os
from flask import Flask
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd


load_dotenv()

openai.api_key  = os.getenv('OPENAI_API_KEY')
#for this you have to set the HUGGINGFACE_TOKEN env variable in your env file
login()
client = OpenAI()

def get_completion(prompt, text,  model="gpt-3.5-turbo"):
    messages = [{"role": "system", "content": prompt},
                {"role":"user", "content": text}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

prompt = """Eres un sistema de detección de texto diseñado para proteger a los niños de posibles amenazas en línea. Tu tarea es clasificar y traducir al español los textos proporcionados por el usuario, que estarán separados entre sí por triple comillas invertidas (```). El contenido se clasificará en una de las siguientes categorías, según los criterios que se indican a continuación.

- OFP: Mensajes con contenido ofensivo dirigido directamente al receptor, como insultos, amenazas o humillaciones.
- OFG: Mensajes con contenido ofensivo dirigido a un grupo de personas de cualquier índole, como discriminación o odio.
- NO: Texto no ofensivo ni peligroso.
- NOE: No ofensivo ni peligroso, pero explícito en su contenido, como lenguaje sexual o violencia.
- GP: Posible grooming o acoso sexual hacia el infante.

Es muy importante que tengas en consideración al clasificar que el objetivo de esta clasificación es proteger a niños.
El formato de entrega debe ser en JSON, encerrado entre un signo menor y mayor <de esta forma> se encuentra la descripción de cada campo. A continuación el formato:

{
    "count": <número de mensajes separados por triple comilla invertida>,
    "classes": <las clases como un arreglo de string, donde cada clase es un string por separado>,
    "translations": <las traducciones al español de las entradas separadas por triple comilla invertida, en un arreglo de strings con cada traducción como un string por separado>
}
    
    Ejemplo de interacción:
    
    usuario: Hi how are you?```Women should stay in the kitchen```gtfo fat fuck!
    
    gpt: {
        "count":3,
        "classes": [
            "NO", 
            "OFG",
            "OFP"
            ],
        "translations":[
            "Hola ¿Cómo estás?",
            "Las mujeres deberían permanecer en la cocina",
            "¡Sal de aqui tipejo gordo!"
        ]    
    }
    
    Ejemplos incorrectos de interacción:
    
    usuario: Hi how are you?```Women "should" stay in the kitchen```gtfo fat fuck!
    
    gpt: {
        "count": 3,
        "classes": [
            "NO", 
            "OFG",
            "OFP"
            ],
        "translations":[
            "Hola ¿Cómo estás?",
            "Las mujeres "deberían" permanecer en la cocina",
            "¡Sal de aqui tipejo gordo!"
        ]    
    }
    
    Esta respuesta es incorrecta puesto que al usar las mismas comillas (") que delimitan el string, se parte el string en dos. Al traducir no puedes
    entregar respuestas que contengan comillas (") dentro de las traducciones. JAMÁS debes entregar respuestas como esta.
    
    usuario: Hi how are you?```Women should stay in the kitchen```gtfo fat fuck!
    
    gpt: {
        "count": 3,
        "classes": [
            "NO", 
            "OFG",
            "OFP",
            "NO"
            ],
        "translations":[
            "Hola ¿Cómo estás?",
            "Las mujeres deberían permanecer en la cocina",
            "¡Sal de aqui tipejo gordo!"
        ]    
    }
    
    El número de traducciones no coincide con el número de clases, ni con el número de oraciones contadas. Es importante
    que coincidan puesto que estos datos serán usados en un dataset.
    
    usuario: Hi how are you?```Women should stay in the kitchen```gtfo fat fuck!
    
    gpt: {
        "count": 5,
        "classes": [
            "NO", 
            "OFG",
            "OFP"
            ],
        "translations":[
            "Hola ¿Cómo estás?",
            "Las mujeres deberían permanecer en la cocina",
            "¡Sal de aqui tipejo gordo!"
        ]    
    }
    El número no coincide con la cantidad real, por lo que esta respuesta es incorrecta.
    
    usuario: Hi how are you?```A woman in the kitchen -as she should-```gtfo fat fuck!
    
    gpt: {
        "count": 3,
        "classes": [
            "NO", 
            "OFG",
            "OFP",
            "NO"
            ],
        "translations":[
            "Hola ¿Cómo estás?",
            "Una mujer en la cocina "como debería estar".",
            "¡Sal de aqui tipejo gordo!"
        ]    
    }
    
    Esta respuesta es incorrecta dado que reemplaza un caracter para-verbal (en este caso -) por las comillas ("). 
    
    Siempre te entregarán tres mensajes o menos, debes asegurarte que el número de clases entregado en el JSON coincida con el número de traducciones. 
    Se especialmente cuidadoso al momento de generar los strings de las traducciones, cambia las comillas (") dentro del string de las traducciones por comillas simples('), tu respuesta será procesada posteriormente y es imperativo que no hayan comillas dentro de los strings. 
    Tu respuesta debe ser coherente con el formato JSON especificado.¡Tú puedes!
    """
    
offendEs = load_dataset("fmplaza/offendes")
sexismreddit = load_dataset("natural-lang-processing/sexismreddit")

pd_sexism_train = pd.DataFrame(sexismreddit['train'])
pd_sexism_train_text = pd_sexism_train[['text', 'label_vector']].copy()
pd_sexism_train_text = pd_sexism_train_text.drop(pd_sexism_train_text[pd_sexism_train_text['label_vector']=='none'].index)
pd_sexism_train_text.replace('"', "'", regex=True, inplace=True)
pd_sexism_train_text.replace('“', "'", regex=True, inplace=True)
pd_sexism_train_text.replace('”', "'", regex=True, inplace=True)
X= pd_sexism_train_text['text']
y = pd_sexism_train_text['label_vector']

print("valores de entrenamiento\n")
print(y.value_counts(), "\n")

pandas_offendES_train = pd.DataFrame(offendEs['train'])
pandas_offendES_validation = pd.DataFrame(offendEs['validation'])
pandas_offendES_test = pd.DataFrame(offendEs['test'])

new_offendEs_train = pandas_offendES_train[['comment', 'label']].copy()
new_offendEs_train = new_offendEs_train.rename(columns={'comment':'text'})
print(new_offendEs_train.head())

new_offendEs_validation = pandas_offendES_validation[['comment', 'label']].copy()
new_offendEs_validation = new_offendEs_validation.rename(columns={'comment':'text'})

new_offendEs_test = pandas_offendES_test[['comment', 'label']].copy()
new_offendEs_test = new_offendEs_test.rename(columns={'comment':'text'})

