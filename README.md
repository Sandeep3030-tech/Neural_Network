# Neural_Network
I am building a simple neural network to demonstrate how a basic model is structured and what it includes. 
This will involve preparing the data, designing the model architecture, training it, evaluating its performance and predicting on new values.

Before implementing any algorithm, it is crucial to visualize the dataset using graphs. 
This helps identify patterns,and understand relationships between variables so that an informed 
decisions can be made and select the most suitable model for the task.
Although I am not doing this in my current algorithm,
it is an important step in building a good machine learning model and should not be skipped.

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
