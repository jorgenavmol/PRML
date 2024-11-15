import numpy as np
import os
import random
import pandas as pd

def split_data(path):
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if len(files) < 200:
        raise ValueError(f"Not enough files to split into training and test sets. Found {len(files)}, but need 200.")
    
    # Assuming filenames contain the digit they represent, e.g., 'digit_0.csv', 'digit_1.csv', etc.
    digit_files = {str(i): [] for i in range(10)}
    for f in files:
        digit = f.split('_')[1].split('.')[0]
        if digit in digit_files:
            digit_files[digit].append(f)
    
    train_files = []
    test_files = []
    
    for digit, files in digit_files.items():
        if len(files) < 20:
            raise ValueError(f"Not enough files for digit {digit}. Found {len(files)}, but need at least 20.")
        random.shuffle(files)
        train_files.extend(files[:10])
        test_files.extend(files[10:20])
    
    return train_files, test_files

def load_data(files, path):
    data = []
    labels = []
    for filename in files:
        df = pd.read_csv(os.path.join(path, filename), header=None)
        data.append(df.iloc[:, :-1].values)
        labels.append(df.iloc[:, -1].values)
    return np.vstack(data), np.hstack(labels)

def extract_features(sample):
    return sample.flatten()

def train_classifier(data, labels):
    classes = np.unique(labels)
    class_means = {cls: np.mean(data[labels == cls], axis=0) for cls in classes}
    return class_means

def digit_classify(sample, class_means):
    sample_features = extract_features(sample)
    distances = {cls: np.linalg.norm(sample_features - mean_vector) for cls, mean_vector in class_means.items()}
    return min(distances, key=distances.get)

def main():
    training_data_path = 'digits_3d/training_data'
    train_files, test_files = split_data(training_data_path)
    
    train_data, train_labels = load_data(train_files, training_data_path)
    test_data, test_labels = load_data(test_files, training_data_path)
    
    class_means = train_classifier(train_data, train_labels)
    
    correct_predictions = sum(digit_classify(test_data[i], class_means) == test_labels[i] for i in range(len(test_data)))
    
    accuracy = correct_predictions / len(test_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
