import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Function to calculate euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def format_data(train_data, train_y, classification_data, classification_y, train_columns, classification_columns):
    # Reconstruct DataFrames with class column
    train_data_scaled = pd.DataFrame(train_data, columns=train_columns)
    train_data_scaled['class'] = train_y

    classification_data_scaled = pd.DataFrame(classification_data, columns=classification_columns)
    classification_data_scaled['class'] = classification_y
    
    return train_data_scaled, classification_data_scaled


def knn_classifier(train_data, classification_data, k=3):
    predictions = []
    neighbor_counts = []
    
    # Iterate over classification data
    for i, test_instance in classification_data.iterrows():
        
        # Calculate the distances for all the training points for the current classification point
        distances = []
        for _, train_instance in train_data.iterrows():
            distance = euclidean_distance(test_instance.drop('class').values, train_instance.drop('class').values)
            distances.append((distance, train_instance['class']))
        
        # Order distances by closest first
        distances.sort(key=lambda x: x[0])
        
        # Obtain k closest neighbors
        neighbors = distances[:k]
        
        # Count the most common classes
        class_votes = {}
        for _, neighbor_class in neighbors:
            class_votes[neighbor_class] = class_votes.get(neighbor_class, 0) + 1
        
        # Obtain the class with the most votes
        predicted_class = max(class_votes, key=class_votes.get)
        predictions.append(predicted_class)
        
        # Store the counts of neighbors per class
        neighbor_counts.append(class_votes)

    return predictions, neighbor_counts


# Function to calculate accuracy of the algorithm
def evaluate_accuracy(predictions, actual_classes):
    correct = sum(pred == actual for pred, actual in zip(predictions, actual_classes))
    accuracy = (correct / len(actual_classes)) * 100
    return accuracy


def run_knn(train_data, classification_data, k, normalized=False):
    # Run KNN with the user-defined k value
    predictions, neighbor_counts = knn_classifier(train_data, classification_data, k)

    # Evaluate accuracy
    accuracy = evaluate_accuracy(predictions, classification_data_scaled['class'])
    normalized_str = "normalized" if normalized else "not normalized"
    print(f"Accuracy ({normalized_str}): {accuracy}%")

    # Prepare the data for neighbor count CSV
    neighbor_count_data = []
    for i, counts in enumerate(neighbor_counts):
        neighbor_count_data.append({
            'Instance': i + 1,
            'tested_negative': counts.get('tested_negative', 0),
            'tested_positive': counts.get('tested_positive', 0),
            'predicted_class': predictions[i]
        })

    # Convert neighbor count data to DataFrame
    neighbor_count_df = pd.DataFrame(neighbor_count_data)

    # Save neighbor count CSV
    normalized_str_filename = "_normalized" if normalized else ""
    neighbor_count_df.to_csv(f"knn_neighbor_counts_k{k}"+normalized_str_filename+".csv", index=False)


# Load files
train_file = "Diabetes-Entrenamiento.csv"
classification_file = "Diabetes-Clasificacion.csv"

train_data = pd.read_csv(train_file)
classification_data = pd.read_csv(classification_file)

# Separate features and labels
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_classification = classification_data.drop('class', axis=1)
y_classification = classification_data['class']

# Normalize using MinMaxScaler: fit on training data, transform both
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_classification_scaled = scaler.transform(X_classification)

# Get k value from user input
k = int(input('Enter the value of k: '))

# Execute knn processes
train_data_scaled, classification_data_scaled = format_data(
    X_train, y_train.values,
    X_classification, y_classification.values,
    X_train.columns, X_classification.columns)
normalized_train_data_scaled, normalized_classification_data_scaled = format_data(
    X_train_scaled, y_train.values,
    X_classification_scaled, y_classification.values,
    X_train.columns, X_classification.columns)

run_knn(train_data_scaled, classification_data_scaled, k)
run_knn(normalized_train_data_scaled, normalized_classification_data_scaled, k, normalized=True)


