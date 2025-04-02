import numpy as np
import pandas as pd

# Function to calculate euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Load files
train_file = "Diabetes-Entrenamiento.csv"
classification_file = "Diabetes-Clasificacion.csv"

train_data = pd.read_csv(train_file)
classification_data = pd.read_csv(classification_file)

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
            if neighbor_class in class_votes:
                class_votes[neighbor_class] += 1
            else:
                class_votes[neighbor_class] = 1
        
        # Obtain the class with the most votes
        predicted_class = max(class_votes, key=class_votes.get)
        predictions.append(predicted_class)
        
        # Store the counts of neighbors per class
        neighbor_counts.append(class_votes)

    return predictions, neighbor_counts

# Function to calculate accuracy of the algorithm
def evaluate_accuracy(predictions, actual_classes):
    correct = 0
    # Loop through predictions and actual_classes
    for pred, actual in zip(predictions, actual_classes):
        if pred == actual:
            correct += 1
    
    accuracy = (correct / len(actual_classes)) * 100
    return accuracy

# Get k value from user input
k = int(input('Enter the value of k: '))

# Run KNN with the user-defined k value
predictions, neighbor_counts = knn_classifier(train_data, classification_data, k)

# Evaluate accuracy
accuracy = evaluate_accuracy(predictions, classification_data['class'])
print(f"Accuracy: {accuracy}%")

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
neighbor_count_df.to_csv(f"knn_neighbor_counts_k{k}.csv", index=False)