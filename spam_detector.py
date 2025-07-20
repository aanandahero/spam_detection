# spam_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=["label", "message"])

# Encode labels
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.message, df.label_num, test_size=0.2, random_state=42)

# Vectorize text
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# Train model
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# Predict and evaluate
y_pred = nb.predict(X_test_dtm)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Test on a new message
msg = ["Free entry in 2 a wkly comp to win FA Cup final tickets"]
msg_dtm = vect.transform(msg)
print("Prediction:", "Spam" if nb.predict(msg_dtm)[0] else "Ham")
