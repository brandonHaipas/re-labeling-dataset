import utils
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

dataset_len = utils.X.size

i = 0
step = 3
dataset_dict = {'text':[], 'label':[] }
while i < dataset_len:
    print("índice: ",i, "\n")
    end = i+step
    if i+step >= dataset_len:
        end = dataset_len
    text = "```".join(utils.X[i:end])
    print(text) 
    completion = utils.get_completion(utils.prompt, text)   
    print(completion)
    try:
        json_obj = json.loads(completion)
    except ValueError as e:
        print("value Error! gpt got the json wrong")
        i+=step
        continue
        
    #add classes to classes array
    largo_clases = len(json_obj['classes'])
    largo_traduciones = len(json_obj['translations'])
    print("largo de clases ", largo_clases)
    print("largo de traducciones ", largo_traduciones)
    if largo_clases != largo_traduciones :
        i+=step
        continue
    new_classes = json_obj['classes']
    dataset_dict['label'].extend(new_classes)
    #add translations to translations array
    new_translations = json_obj['translations']
    dataset_dict['text'].extend(new_translations)
    i+=step
    
    
sexism_dataframe = pd.DataFrame(dataset_dict)
grooming_data = sexism_dataframe[sexism_dataframe['label']=='GP']
grooming_data.reset_index(drop=True, inplace=True)
grooming_data.to_excel("grooming.xlsx")
grooming_data.to_csv("grooming.csv")

# I have to change the label form 'GP' to 4, since the labels in offendes go from 0 to 3 ..

grooming_data.loc[grooming_data.label =='GP', 'label'] = 4

train, test = train_test_split(grooming_data, test_size=0.2)
train, validation = train_test_split(train, test_size= 0.3)

train = train[['text', 'label']].copy()
test = test[['text', 'label']].copy()
validation= validation[['text', 'label']].copy()

train_list = [utils.new_offendEs_train, train]
validation_list = [utils.new_offendEs_validation, validation]
test_list = [utils.new_offendEs_test, test]

train_dataframe = pd.concat(train_list)
validation_dataframe = pd.concat(validation_list)
test_dataframe = pd.concat(test_list)

validation_dataframe= validation_dataframe.reset_index()
train_dataframe = train_dataframe.reset_index()
test_dataframe = test_dataframe.reset_index()

hugg_train = Dataset.from_pandas(train_dataframe)
hugg_validation = Dataset.from_pandas(validation_dataframe)
hugg_test = Dataset.from_pandas(test_dataframe)


new_data_dict = DatasetDict(
    {
        "train": hugg_train,
        "validation": hugg_validation,
        "test": hugg_test,
    }
)

new_data_dict.save_to_disk()
new_data_dict['train'].to_csv("cyberdefender_train.csv")
new_data_dict['validation'].to_csv("cyberdefender_validation.csv")
new_data_dict['test'].to_csv("cyberdefender_test.csv")
new_data_dict['train'].to_json("cyberdefender_train.json")
new_data_dict['validation'].to_json("cyberdefender_validation.json")
new_data_dict['test'].to_json("cyberdefender_test.json")