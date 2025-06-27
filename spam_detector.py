import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('./spam.csv', encoding='latin-1') 
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data = data.rename(columns={'v1':'label', 'v2':'text'})

def preprocess_text(text):
    text = text.lower()

    return text

data['text'] = data['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], 
                                                test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000) 
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

joblib.dump(model, 'spam_model.pkl') 
joblib.dump(vectorizer, 'vectorizer.pkl')