import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

# File reading
fact_ver_dataframe = pd.read_excel('Claims.xlsx')

# Set up BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

# Claim processing
token_label_dataframe = pd.DataFrame(columns=['claim', 'tensor', 'label'])

#arrays for claim tokens and labels
claim_tokens_array = []
claim_label_array = []
# Convert claims to tokens

for index, row in fact_ver_dataframe.iloc[::6].iterrows():
    claim = row[2]
    claim_token = tokenizer.encode(claim, add_special_tokens=True)
    claim_tokens_array.append(claim_token)
    
    claim_label = row[3]
    if claim_label == 'F':
        claim_label = 0
    elif claim_label == 'T':
        claim_label = 1
    elif claim_label == 'N':
        claim_label = 2
    claim_label_array.append(claim_label)

# Convert evidence to tokens
evidence_tokens_array = []
for index, row in fact_ver_dataframe.iterrows():
    evidence = row[5]
    evidence_token = tokenizer.encode(evidence, add_special_tokens=True)
    evidence_tokens_array.append(evidence_token)


input_tensor_array = []
# combined claim and evidence tokens to create a tensor
for i, claim in enumerate(claim_tokens_array):
    input_tokens = claim
    for j in range(6):
        input_tokens.extend(evidence_tokens_array[i * 6 + j])

    # Pad or truncate to a fixed size
    max_length = 512
    input_tokens = input_tokens[:max_length] + [tokenizer.pad_token_id] * (max_length - len(input_tokens))
    input_tensor_array.append(tf.constant([input_tokens]))

# Pass the input tensor array through the model
outputs = model(input_tensor_array[0])
# Get the encoded representations from the model
encoded_representations = outputs.last_hidden_state

# Create a Keras model for classification
classifier_inputs = tf.keras.Input(shape=(512,))
num_classes = 3
softmax_layer = layers.Dense(num_classes, activation='softmax')
predictions = softmax_layer(classifier_inputs)

#Set up the model with classifier
model = keras.Model(inputs=classifier_inputs, outputs=predictions)

# Compile the model with an appropriate loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Preparing training data
x_train, x_val, y_train, y_val = train_test_split(input_tensor_array, claim_label_array, test_size=0.2, random_state=42)

# Convert target values to one-hot encoded arrays
num_classes = 3  # Set the correct number of classes (3 for your case)
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes, dtype='float32')
y_val = keras.utils.to_categorical(y_val, num_classes=num_classes, dtype='float32')

x_train = np.array(x_train)
x_train = tf.squeeze(x_train, axis=1)
x_val = tf.squeeze(x_val, axis=1)
input_shape = model.layers[0].input_shape
# Convert input tensors to NumPy arrays
x_train = np.array(x_train)
x_val = np.array(x_val)

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

# Evaluate model
y_val = keras.utils.to_categorical(y_val, num_classes=num_classes, dtype='float32')
y_val_pred = model.predict(x_val)
y_val_labels = np.argmax(y_val, axis=1)  # Convert one-hot encoded labels back to class labels

loss, accuracy = model.evaluate(x_val, y_val_labels)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Make predictions
test_text = "your test text goes here"
test_input_tokens = tokenizer.encode(test_text, add_special_tokens=True)
test_input_tokens = test_input_tokens[:max_length] + [tokenizer.pad_token_id] * (max_length - len(test_input_tokens))
test_input_tensor = tf.constant([test_input_tokens])
