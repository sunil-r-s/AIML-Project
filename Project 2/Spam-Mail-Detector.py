import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import os

try:
    # Check if file exists first
    file_path = r"AIML-Project//Project 2//spam.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please check the file path and make sure the file exists.")
        exit()
    
    # Load data with error handling
    print("Loading data...")
    df = pd.read_csv(file_path, sep="\t", names=['label', 'text'])
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Check data structure
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Remove any rows with missing values
    df = df.dropna()
    print(f"Shape after removing missing values: {df.shape}")
    
    # Check unique labels
    print(f"\nUnique labels: {df['label'].unique()}")
    
    # Map labels to numerical values
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Check if mapping worked properly
    if df['label'].isnull().any():
        print("Warning: Some labels couldn't be mapped properly!")
        print("Unique values after mapping:", df['label'].unique())
    
    # Check class distribution
    print(f"\nClass distribution:\n{df['label'].value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize the text
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        ngram_range=(1, 2),
        max_features=5000,  # Limit features to prevent overfitting
        lowercase=True,
        strip_accents='unicode'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Train the model
    print("\nTraining model...")
    model = MultinomialNB(alpha=1.0)  # You can tune this parameter
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Print results
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Test with sample messages
    print("\nTesting with sample messages:")
    samp_msg = ["Win a free ticket now!!!", "Let's meet for lunch tomorrow."]
    samp_msg_vec = vectorizer.transform(samp_msg)
    predictions = model.predict(samp_msg_vec)
    probabilities = model.predict_proba(samp_msg_vec)
    
    for i, msg in enumerate(samp_msg):
        result = "Spam" if predictions[i] == 1 else "Ham"
        confidence = max(probabilities[i]) * 100
        print(f"Message: '{msg}'")
        print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
        print()

except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")