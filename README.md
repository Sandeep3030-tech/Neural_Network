# Neural_Network
I am building a simple neural network to demonstrate how a basic model is structured and what it includes. 
This will involve preparing the data, designing the model architecture, training it, evaluating its performance and predicting on new values.

Before implementing any algorithm, it is crucial to visualize the dataset using graphs. 
This helps identify patterns,and understand relationships between variables so that an informed 
decisions can be made and select the most suitable model for the task.
Although I am not doing this in my current algorithm,
it is an important step in building a good machine learning model and should not be skipped.

# How does a Neural Network Work
A neural network is a web of interconnected neurons that pass signals to each other trying to mimic a human brain.

For example: when a kid learns to recognize a cat, 
they might see a few pictures and hear feedback about what a cat is.
At first, the kid might get confused and 
call a dog a cat. Over time, by repeatedly comparing what they see 
with the correct answer, the kid’s brain begins to notice certain patterns.
They might realize that most cats have pointy ears, whiskers, and a particular way 
of meowing. Each time the kid gets corrected or confirms what a cat looks like, 
their brain adjusts its "mental map" building  stronger connections around those 
key features. This ongoing process of trial, feedback, and adjustment makes the kid 
better at recognizing a cat.

In a similar way, a neural network gets better at predicting by constantly updating
its internal parameters (weights and biases) based on the errors it made in previous 
guesses. Just as a kid learns from repeated experiences and corrections, the neural 
network refines its pattern recognition skills through a process of guessing, 
comparing its guess to the real value, learning from the mistake, and then adjusting 
its settings. As the network makes more predictions, it starts to recognize patterns 
in the data — like how higher BMI, older age, or smoking status can affect medical costs.
This ability to detect patterns helps the network make more accurate predictions over
time. This repetition allows both the kid and the network to improve their accuracy 
by learning from mistakes and adapting their understanding of the patterns.

# Loading the Dataset
Using pandas to read the CSV file containing the dataset for features and target values.
Checking for any missing values in the dataset and printing the count of missing values 
for each column.
Also printing the dataset to verify the column names for accurate usage later on.
```python
import pandas as pd

dataset=pd.read_csv('medical_cost.csv')
null_values = dataset.isnull().sum().sort_values(ascending=False)

print(null_values)
dataset
```
# Target data and Features data Extraction
Prompting the user to select the target column.
Once selected, the dataset is divided into two variables:
- 'target' contains the target column.
- 'features' contains all the other columns that will be used as features for training the model.
```python
target_column = input("Please enter the name of the target column: ")
if target_column not in dataset.columns:
    print(f"Error: {target_column} is not a valid column in the dataset.")
else:
    target = dataset[target_column]
    features = dataset.drop(columns=[target_column])

print(target)
print(features)
```
# Converting string values to numerical values
After selecting the target and feature columns,
all non-numerical values are converted to numerical values 
using pandas' `get_dummies()` function. 
This applies one-hot encoding, where each unique category is turned 
into a binary column (1 for presence, 0 for absence) 
since neural networks cannot process string values directly.
```python
if target.dtype == 'object':  
    target = pd.get_dummies(target)

for column in features.columns:
    if features[column].dtype == 'object':
        string_to_num = pd.get_dummies(features[column])
        
        features = features.drop(column, axis=1).join(string_to_num)

print(target.astype(int))
print(features.astype(int))
```
# Scalling the data
Neural networks work best when all input features are on a similar scale. 
If some features have values ranging from 0 to 1000 while others range from 0 to 1, 
the network may give more importance to the larger numbers, leading to inaccurate training.
This happens because neural networks adjust their weights based on the magnitude 
of the inputs, and larger values can dominate the learning process. So  Min-Max Scaling, 
will rescales all features to a fixed range, usually between 0 
and 1. This ensures that every feature contributes equally, making the training process 
more stable and efficient.
```python
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

feature_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(features.astype(float))

dump(feature_scaler, 'scaler.joblib') 
print(features_scaled)
```
# Building a Neural Network
```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#Splitting dataset into training and testing sets, allocating 10% for testing the model. 
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.1, random_state=0)

# Neural Network Model Architecture
model = Sequential([
    Dense(16, activation="relu"),  # Hidden layer 1 (16 neuron)
    Dense(8, activation="relu"),  # Hidden layer 2 (8 neuron)
    Dense(4, activation="relu"),  # Hidden layer 3 (4 neuron)
    Dense(2, activation="relu"),  # Hidden layer 4 (2 neuron)
    
    Dense(1),  # Output layer (1 neuron)
])

#Configuring the model for training by specifying the optimizer and the loss function
model.compile(optimizer="adam",loss="mae")

# Training the Model
history = model.fit(  
    X_train, y_train,
    epochs=50, #The model will reapet training 50 times.        
    batch_size=32, #Model looks at 32 pieces at a time, calculates the error, adjusts its internal settings, and moves on to the next 32 pieces
    validation_split=0.1, #10% of data is used to check how well model is performing
    verbose=1, #Display progress information during training
)


# Plot training vs. validation loss to detect overfitting. 
#If  the training loss keeps decreasing while the validation loss starts increasing or stays constant it is a sign of overfitting.
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.show()



#The model makes predictions using the test data (X_test), 
#Calculates how well the predictions match the actual results (y_test) using the R-squared score.
predictions = model.predict(X_test).flatten()
NN_r2 = r2_score(y_test, predictions) * 100

# Printing Results
print(f"Neural Network R-squared = {NN_r2:.2f}%")

# Save the trained model
model.save("Best_NeuralNetwork.keras")
print("Model saved as 'Best_NeuralNetwork.keras'")
```
# Predicting new values from the trained model
```python
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

#Making predictions using a pre-trained neural network model.
# Load trained model
model = load_model("Best_NeuralNetwork.keras")

#  Load the correct scaler
scaler = joblib.load("scaler.joblib")

# Define input data
input_data = {
    "Age":[58],
    "BMI":[15.6],
    "Children":[2],
    "female":[0],
    "male":[1],
    "no":[0],
    "yes":[1],
    "northeast":[0],
    "northwest":[1],
    "southeast":[0],
    "southwest":[0],
}

# Convert to DataFrame
new_features = pd.DataFrame(input_data)
new_features_scaled = scaler.transform(new_features)
prediction_scaled = model.predict(new_features_scaled, verbose=0)

print(f"Predicted value: {prediction_scaled[0][0]:.2f}")
```
