from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from pickle import load

# Load the saved model
class TabularNN(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dims, num_numerical_features, hidden_size, output_size):
        super(TabularNN, self).__init__()
        self.embeddings = torch.nn.ModuleDict({
            col: torch.nn.Embedding(num_embeddings[col], embedding_dims[col])
            for col in num_embeddings
        })
        total_embedding_size = sum(embedding_dims.values())
        self.total_input_size = total_embedding_size + num_numerical_features
        self.fc1 = torch.nn.Linear(self.total_input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, numerical_features, categorical_features):
        embedded_features = [
            self.embeddings[col](categorical_features[col].long())
            for col in self.embeddings
        ]
        embedded_features = torch.cat(embedded_features, dim=1)
        combined_features = torch.cat([embedded_features, numerical_features], dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return self.softmax(x)

# Define model parameters
num_embeddings = {
    'root_stone': 2, 'root_grate': 2, 'root_other': 2,
    'trunk_wire': 2, 'trnk_light': 2, 'trnk_other': 2,
    'brch_light': 2, 'brch_shoe': 2, 'brch_other': 2,
    'curb_loc': 2, 'sidewalk': 2, 'spc_common': 133, 'nta': 188
}
embedding_dims = {
    'root_stone': 2, 'root_grate': 2, 'root_other': 2,
    'trunk_wire': 2, 'trnk_light': 2, 'trnk_other': 2,
    'brch_light': 2, 'brch_shoe': 2, 'brch_other': 2,
    'curb_loc': 2, 'sidewalk': 2, 'spc_common': 16, 'nta': 16
}
num_numerical_features = 1
hidden_size = 32
output_size = 3

# Initialize the model
model = TabularNN(num_embeddings, embedding_dims, num_numerical_features, hidden_size, output_size)
model.load_state_dict(torch.load('tabularnn_model.pth'))
model.eval()

# Load preprocessing objects (e.g., scaler, label encoders)
# Replace with your actual preprocessing objects
scaler = load(open('scaler.plk', 'rb'))
label_encoders = load(open('/encoder.plk', 'rb'))
enc = {'health': LabelEncoder()}
enc['health'].classes_ = np.array(['Fair', 'Good', 'Poor'])

# Define input data model
class InputData(BaseModel):
    root_stone: str
    root_grade: str
    root_other: str
    trunk_wire: str
    trnk_ligth: str
    trnk_other: str
    brch_light: str
    brch_shoe: str
    brch_other: str
    curb_loc: str
    sidewalk: str
    spc_common: str
    nta: str
    tree_dbh: int

# Initialize FastAPI app
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Preprocess categorical features
        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Preprocess numerical features
        numerical_features = scaler.transform(input_df[['tree_dbh']])
        numerical_features = torch.tensor(numerical_features, dtype=torch.float32)

        # Preprocess categorical features
        categorical_features = {
            col: torch.tensor(input_df[col].values, dtype=torch.float32)
            for col in label_encoders
        }

        # Make prediction
        with torch.no_grad():
            prediction = model(numerical_features, categorical_features)
            _, pred = torch.max(prediction, 1)
            pred = enc.inverse_transform(pred)

        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "TabularNN Model API"}