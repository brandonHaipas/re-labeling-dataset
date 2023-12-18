
import openai
from openai import OpenAI
import os
from flask import Flask
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

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

prompt = """Eres un detector de texto potencialmente peligroso para niños. 
    Tienes que clasificar los textos entregados por el usuario que estarán separados por doble backticks en 
    las siguientes categorias:
    -OFP: Mensajes con contenido ofensivo dirigido directo al receptor.
    -OFG: Mensajes con contenido ofensivo a un grupo de personas de cualquier índole.
    -NO: Texto no ofensivo ni peligroso.
    -NOE: No ofensivo ni peligroso, pero explícito en su contenido.
    -GP: Posible grooming o acoso sexual hacia el infante.

    El formato de entrega es json, de la siguiente manera, lo que debes entregar estará entre un signo menor y uno mayor <de esta forma>:
    
    {
        "classes": <las clases separadas por comas y como string>,
        "translations": <las traducciones al español de cada una de las entradas separadas por doble backtics>
    }
    
    Ejemplo de interacción:
    
    user: Hi how are you?``Women should stay in the kitchen
    
    gpt: {
        "classes": [
            "NO", 
            "OFG"
            ],
        "translations":[
            "Hola ¿Cómo estás?",
            "Las mujeres deberían permanecer en la cocina"
        ]    
    }
    """
    
offendEs = load_dataset("fmplaza/offendes")
sexismreddit = load_dataset("natural-lang-processing/sexismreddit")

pd_sexism_train = pd.DataFrame(sexismreddit['train'])
pd_sexism_train_text = pd_sexism_train[['text', 'label_vector']].copy()
X = pd_sexism_train_text['text']
y = pd_sexism_train_text['label_vector']

X_train, X_test,y_train,y_test = train_test_split(X, y, stratify=y, test_size=0.5)
print("valores de entrenamiento\n")
print(y_train.value_counts(), "\n")
print("valores de test\n")
print(y_test.value_counts(), "\n")
pandas_offendES_train = pd.DataFrame(offendEs['train'])
new_offendEs_train = pandas_offendES_train[['comment', 'label']].copy()
print(new_offendEs_train.sample(5))