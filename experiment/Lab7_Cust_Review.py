# =====================================================
# Program: Sentiment Analysis using Naïve Bayes
# Dataset: Sample Customer Product Reviews (can be replaced with Amazon reviews)
# Objective: Classify sentiment as Positive, Negative, or Neutral
# Use a dataset of customer product reviews (e.g., Amazon reviews) to classify the sentiment
# of each review as positive, negative, or neutral using a pre-trained machine learning model
# (e.g., Naïve Bayes). Evaluate the accuracy of your sentiment classifier.
# =====================================================

# Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Step 2: Create or Load the Dataset
data = 'reviews150.csv'

df = pd.read_csv(data)
print("Sample Dataset:\n")
df = df.iloc[:, 1:]
print(df.head())

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, random_state=42,
stratify=df['sentiment']   # ensures balanced classes
)

# Step 4: Convert Text Data into Numerical Features using Bag of Words
# vectorizer = CountVectorizer()

# Step 4a: Removed stop words from English lang
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)



# Step 5: Train the Naïve Bayes Classifier
# model = MultinomialNB()

# Step 5a: train by Logistic Regression
# model = LogisticRegression(max_iter=1000)

# Step 5b: train by the model SVM
model = LinearSVC()
model.fit(X_train_counts, y_train)

# Step 6: Predict Sentiments on Test Data
y_pred = model.predict(X_test_counts)

# Step 7: Evaluate the Model
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred,  zero_division=0))

# Step 8: Test with a New Review
new_review = ["The product is really good and works perfectly!"]
new_review_count = vectorizer.transform(new_review)
prediction = model.predict(new_review_count)
print("\nTest Review:", new_review[0])
print("Predicted Sentiment:", prediction[0])
