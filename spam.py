import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('spam.csv', encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'text']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['text']  # email text
y = data['label']  # label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_counts, y_train)
y_pred = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

def classify_email(email_text):
    email_count = vectorizer.transform([email_text])
    prediction = model.predict(email_count)
    return "Spam" if prediction[0] == 1 else "Not Spam"

sample_email = "Congratulations! You've won a free iPhone. Click here to claim your prize!"
print(f"\nSample email classification: {classify_email(sample_email)}")