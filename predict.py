import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from elm import ELM

# Load the diabetes dataset
diab = pd.read_csv("diabetes.csv")

# Split the data into features (X) and target (y)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Fit the scaler to the training data
scaler.fit(diab.drop(columns=['Outcome']))

# Save the fitted scaler back to the pickle file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#diab = pd.read_csv("diabetes.csv")
#X = diab.drop('Outcome', axis=1)
X=diab
y = diab['Outcome']
X = np.array(X)
y = np.array(y)
Z=diab
    
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# scaler1=StandardScaler()
# X_scale=scaler1.fit_transform(Z)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=32)

# Initialize and train the ELM classifier with population size
elm_classifier = ELM(hidden_units=250, population_size=50, activation_fn='sigmoid', x=X_train, y=y_train)
beta, train_score = elm_classifier.fit()
print("Training Accuracy:", train_score * 100)

# Evaluate the model on the test set
test_score = elm_classifier.score(X_test, y_test)
print("Test Accuracy:", test_score * 100)

# Save ELM model (optional)
with open('elm_model.pkl', 'wb') as f:
    pickle.dump(elm_classifier, f)

# Save PCA model
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
