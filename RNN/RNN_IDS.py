#!/usr/bin/env python3

import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score, root_mean_squared_error, mean_squared_error, mean_absolute_error, precision_score, confusion_matrix, multilabel_confusion_matrix, classification_report, f1_score, precision_score, recall_score, log_loss

# Load data
def load_data(filepath):
    monday = pd.read_csv(f'{filepath}/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv', sep=",", encoding='utf-8')
    tuesday = pd.read_csv(f'{filepath}/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv', sep=",", encoding='utf-8')
    wednesday = pd.read_csv(f'{filepath}/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv', sep=",", encoding='utf-8')
    thursday_morning = pd.read_csv(f'{filepath}/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', sep=",", encoding='utf-8')
    thursday_afternoon = pd.read_csv(f'{filepath}/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', sep=",", encoding='utf-8')
    friday_ddos = pd.read_csv(f'{filepath}/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', sep=",", encoding='utf-8')
    friday_pcap = pd.read_csv(f'{filepath}/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', sep=",", encoding='utf-8')
    friday_morning = pd.read_csv(f'{filepath}/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv', sep=",", encoding='utf-8')

    dataframes = [monday, tuesday, wednesday, thursday_morning, thursday_afternoon, friday_ddos, friday_morning, friday_pcap]
    data = pd.concat(dataframes)
    data = data.replace([-np.inf, np.inf], np.nan)
    data.dropna(inplace=True)

    return data

# Preprocess data
def preprocess_data(data):
    # Drop unnecessary columns (e.g., ID or timestamp if present)
    data = data.drop(columns=[' Flow ID', ' Source IP', ' Destination IP', ' Timestamp'], errors='ignore')

    # Handle missing values
    data = data.fillna(0)

    # Encode labels
    label_encoder = LabelEncoder()
    data[' Label'] = label_encoder.fit_transform(data[' Label'])
    num_classes = len(label_encoder.classes_)

    # Normalize numerical features
    scaler = StandardScaler()
    features = data.drop(columns=[' Label']).values
    labels = data[' Label'].values
    features = scaler.fit_transform(features)

    # Principal component analysis
    num_components = 10
    pca = PCA(n_components=num_components)
    features_pca = pca.fit_transform(features)
    feature_selector = SelectKBest(score_func=f_classif, k='all')
    features = feature_selector.fit_transform(features_pca, labels)

    return features, labels, num_classes

# Prepare sequential data
def create_sequences(features, labels, seq_length):
    sequences, seq_labels = [], []
    for i in range(len(features) - seq_length):
        sequences.append(features[i:i + seq_length])
        seq_labels.append(labels[i + seq_length - 1])  # Use the last label in the sequence
    return np.array(sequences), np.array(seq_labels)

# Step 4: Build the RNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def metrics(X_test, y_test, y_pred):
    print("Computing accuracy score")
    _, accuracy = model.evaluate(X_test, y_test)
    print("Computing Categorical cross-entropy")
    cce = categorical_crossentropy(y_test, y_pred)
    cce.numpy()
    cce = sum(cce)/len(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print("Computing precision score")
    precision = precision_score(y_true=y_test, y_pred=y_pred, average="weighted", zero_division=0.0)
    print("Computing f1 score")
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
    print("Computing recall score")
    recall = recall_score(y_true=y_test, y_pred=y_pred, average="weighted")
    print("Computing MAE score")
    mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
    print("Computing MSE score")
    mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
    print("Computing RMSE score")
    rmse = root_mean_squared_error(y_true=y_test,y_pred=y_pred)
    return accuracy, precision, f1, recall, mae, mse, rmse, cce

# Main script
if __name__ == "__main__":
    # Load data
    filepath = '../data/CIC_IDS_2017'
    data = load_data(filepath)

    # Preprocess data
    features, labels, num_classes = preprocess_data(data)

    # Create sequences
    seq_length = 20
    X, y = create_sequences(features, labels, seq_length)
    y = to_categorical(y, num_classes=num_classes)  # Convert labels to one-hot encoding

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build model
    model = build_model(input_shape=(seq_length, X.shape[2]), num_classes=num_classes)

    # Model summary
    model.summary()
    plot_model(model, "model.png")

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8192,
        validation_data=(X_val, y_val)
    )

    # Evaluate model
    y_pred = model.predict(X_test)

    accuracy, precision, f1, recall, mae, mse, rmse, cce = metrics(X_test, y_test, y_pred)
    results = PrettyTable(["Metric", "Result"])
    results.add_row(["Accuracy", f"{accuracy * 100:.2f}%"])
    results.add_row(["Precision", f"{precision}"])
    results.add_row(["F1", f"{f1}"])
    results.add_row(["Recall", f"{recall}"])
    results.add_row(["MAE", f"{mae}"])
    results.add_row(["MSE", f"{mse}"])
    results.add_row(["RMSE", f"{rmse}"])
    results.add_row(["Log loss", f"{cce}"])
    print(results)

    with open("metrics.txt", "w") as file:
        file.write(str(results))

    # Save model
    #model.save('rnn_ids_model.h5')
