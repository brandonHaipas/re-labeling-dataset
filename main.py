import utils
import json
import pandas as pd

dataset_len = utils.X_train.size

i = 0
dataset_dict = {'text':[], 'label':[] }
while i < dataset_len:
    end = i+10
    if i+10 >= dataset_len:
        end = dataset_len
    text = "``".join(utils.X_train[i:end]) 
    completion = utils.get_completion(utils.prompt, text)   
    print(completion)
    json_obj = json.loads(completion)
    #add classes to classes array
    new_classes = json_obj['classes']
    dataset_dict['label'].extend(new_classes)
    #add translations to translations array
    new_translations = json_obj['translations']
    dataset_dict['text']
    i+=10
    
    
sexism_dataframe = pd.DataFrame(dataset_dict)