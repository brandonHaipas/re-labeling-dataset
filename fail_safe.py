import utils
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

grooming_dataframe = pd.read_excel('grooming.xlsx', index_col=0)
grooming_dataframe.reset_index(drop=True, inplace=True)
grooming_dataframe.loc[grooming_dataframe.label =='GP', 'label'] = 4

train, test = train_test_split(grooming_dataframe, test_size=0.2)
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
print(train_dataframe.head())
validation_dataframe= validation_dataframe.reset_index()
train_dataframe = train_dataframe.reset_index()
test_dataframe = test_dataframe.reset_index()
hugg_train = Dataset.from_pandas(train_dataframe)
hugg_validation = Dataset.from_pandas(validation_dataframe)
hugg_test = Dataset.from_pandas(test_dataframe)

#now we reset the index

new_data_dict = DatasetDict(
    {
        "train": hugg_train,
        "validation": hugg_validation,
        "test": hugg_test,
    }
)

print(new_data_dict)
new_data_dict.save_to_disk("cyberdefender.hf")
new_data_dict['train'].to_csv("train.csv")
new_data_dict['validation'].to_csv("validation.csv")
new_data_dict['test'].to_csv("test.csv")
new_data_dict['train'].to_csv("train.json")
new_data_dict['validation'].to_csv("validation.json")
new_data_dict['test'].to_csv("test.json")