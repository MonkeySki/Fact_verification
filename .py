import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)


# File reading
fact_ver_dataframe = pd.read_excel('C:/Users/breco/Documents/Code/Fact_verification/Claims.xlsx')

# Set up BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

# Claim processing
token_label_dataframe = pd.DataFrame(columns=['claim','tensor','label'])

# Convert claims to tokens
claim_tokens_array =[]
for index, row in fact_ver_dataframe.iloc[::6].iterrows():
    claim = row[2]
    claim_token = tokenizer.encode(claim, add_special_tokens=True)
    claim_tokens_array.append(claim_token)

# Convert evidence to tokens
evidence_tokens_array = [] 
for index, row in fact_ver_dataframe.iterrows():
    evidence = row[5]
    evidence_token = tokenizer.encode(evidence,add_special_tokens = True)
    evidence_tokens_array.append(evidence_token)


input_tensor_array = []
# combined claim and evidence tokens to create a tensor
for i,claim in enumerate(claim_tokens_array):
    input_tokens = claim
    for j in range(6):
        input_tokens.extend(evidence_tokens_array[i*6+j])
    
    # Pad or truncate to a fixed size

    print(input_tokens)
    max_length = 512
    input_tokens = input_tokens[:max_length] + [tokenizer.pad_token_id] * (max_length - len(input_tokens))
    print()
    print(input_tokens)
    input_tensor_array.append(tf.constant([input_tokens]))

print(len(input_tensor_array))
print(len(claim_tokens_array))
for index, row in fact_ver_dataframe.iterrows():
    
    # Convert a claim and it's 6 pieces of evidence into tokens
    claim = row[2]
    #each claim has 6 pieces of evidence
    evidence_1 = row[5]
    evidence_2 = "Evidence 2 text goes here."
    # Tokenize claim and evidence texts
    claim_tokens = tokenizer.encode(claim, add_special_tokens=True)
    evidence_1_tokens = tokenizer.encode(evidence_1, add_special_tokens=True)
    evidence_2_tokens = tokenizer.encode(evidence_2, add_special_tokens=True)


    # Tokenize claim and evidence tokens
    input_tokens = claim_tokens + evidence_1_tokens + evidence_2_tokens

    # Pad or truncate to a fixed size
    max_length = 512
    input_tokens = input_tokens[:max_length] + [tokenizer.pad_token_id] * (max_length - len(input_tokens))

    # Convert input_tokens to a tensor
    # input_tensor = tf.convert_to_tensor([input_tokens])
    input_tensor = tf.constant([input_tokens])

    # Add the input tensor and corresponding label/reason to dataframe
    # token_label_dataframe.append()
