import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

from visualizer import plot_vectors_to_pdf

from transformers import AutoTokenizer, BertModel, BertForMaskedLM
tokenizer = AutoTokenizer.from_pretrained('dicta-il/BEREL_2.0')
model = BertModel.from_pretrained('dicta-il/BEREL_2.0')
# model = BertForMaskedLM.from_pretrained('dicta-il/BEREL_2.0')

# tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
# model = BertModel.from_pretrained("avichr/heBERT")

def generate_vectors(df, name_column, content_column, vectors_file, progress_signal=None):
    vectors = {}
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        inputs = tokenizer(row[content_column], return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        # Take the mean of the last hidden state to get a single vector that represents the entire text
        mean_vector = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy().flatten().tolist()
        vectors[row[name_column]] = mean_vector
        if progress_signal:
            progress_signal.emit(i + 1)
    if progress_signal:
        progress_signal.emit(100)
    # Save vectors as a pickle file
    with open(vectors_file, 'wb') as f:
        pickle.dump(vectors, f)

def classify_vector(df, content_column, label_column):
    df.dropna(subset=[content_column, label_column], inplace=True)

    vectors = []
    labels = []
    for _, row in df.iterrows():
        class_name = row[label_column]
        inputs = tokenizer(row[content_column], return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        cls_token = outputs[0][:, 0, :].detach().numpy().flatten().tolist()
        vectors.append(cls_token)
        labels.append(class_name)

    return vectors, labels

def train_classifier(vectors, labels, Classifier):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42)

    # Train the classifier
    classifier = Classifier()
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

    return classifier


def train_multiple_classifiers(vectors_file):
    classifiers = [LogisticRegression, SVC, GaussianNB, RandomForestClassifier, GradientBoostingClassifier]

    # Load vectors from a pickle file
    with open(vectors_file, 'rb') as f:
        vectors_dict = pickle.load(f)

    # Extract vectors and labels from the dictionary
    labels = [key.split('_')[-1] for key in vectors_dict.keys()]
    vectors = list(vectors_dict.values())

    # Prepare a list to store the results and a PdfPages object to store the plots
    results = []
    pdf_pages = PdfPages('classifier_plots.pdf')

    # Create a KFold object for 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Train each classifier, store the results, and plot the 2D vectors
    for Classifier in tqdm(classifiers):
        fold_results = []
        for train_index, test_index in kf.split(vectors):
            X_train, X_test = np.array(vectors)[train_index], np.array(vectors)[test_index]
            y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

            classifier = Classifier()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            # Calculate metrics
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)

            metrics = {
                'Classifier': Classifier.__name__,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Accuracy': accuracy
            }

            # Store the results
            fold_results.append(metrics)

            # Plot the 2D vectors
            plot_vectors_to_pdf(X_test, y_pred, Classifier.__name__, metrics, pdf_pages)

        # Average the metrics over all folds and store the results
        numeric_keys = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
        avg_metrics = {key: np.mean([metrics[key] for metrics in fold_results]) for key in numeric_keys}
        avg_metrics['Classifier'] = fold_results[0]['Classifier']
        results.append(avg_metrics)

        # Save the trained classifier
        with open(f'classifiers/{Classifier.__name__}_classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)

    # Close the PdfPages object
    pdf_pages.close()

    # Convert the results to a DataFrame and save it as a CSV file
    df = pd.DataFrame(results)
    df.to_csv('classifier_metrics.csv', index=False)

def classify_document(single_document, classifier):
    inputs = tokenizer(single_document, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    single_doc_vector = outputs[0][:, 0, :].detach().numpy().flatten().tolist()

    # Predict the class of the single document
    predicted_class = classifier.predict([single_doc_vector])[0]

    return predicted_class

def compare_with_documents(single_document, class_vectors):
    # Tokenize and encode the single document
    inputs = tokenizer(single_document, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    single_doc_vector = outputs[0][:, 0, :].detach().numpy().flatten().tolist()

    # Calculate cosine similarity with each class of documentsאה
    similarities = {}
    for class_name, vectors in class_vectors.items():
        class_vectors_np = np.array([vec for _, vec in vectors])
        similarity = cosine_similarity([single_doc_vector], class_vectors_np).mean()
        similarities[class_name] = similarity

    # Find the class with the highest average similarity
    best_class = max(similarities, key=similarities.get)

    return best_class

def get_saved_classifier(classifier_name):
    with open(f'classifiers/{classifier_name}_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    return classifier