Fake News Detection using Machine Learning
🔍 Overview
This project aims to detect whether a news article is fake or real using machine learning. We use a labeled dataset of real and fake news articles and apply text preprocessing, TF-IDF vectorization, and Logistic Regression for classification.

📁 Dataset
We use two CSV files:

Fake.csv — contains fake news articles

True.csv — contains real news articles

These files are combined and labeled to create a training dataset.

📌 Dataset Source: Kaggle - Fake and Real News Dataset

📌 Technologies Used
Python 🐍

Scikit-learn

Pandas

NumPy

TF-IDF Vectorizer

Logistic Regression

⚙️ How it Works
Data Loading: Read both datasets (Fake.csv and True.csv)

Labeling: Assign labels → Fake: 1, Real: 0

Combining & Shuffling: Merge and randomize the rows

Preprocessing: Convert text data to numeric vectors using TF-IDF

Train-Test Split: Split data into training and testing sets

Model Training: Train using Logistic Regression

Evaluation: Measure accuracy, precision, recall, etc.

Custom Prediction: Test with your own input sentence

🧪 Example
python
Copy
Edit
sample = ["Education is necessary for nation building."]
vector = vectorizer.transform(sample)
print("Prediction:", model.predict(vector))
✅ Results
Accuracy: ~92% on the test set (may vary slightly)

Handles short or long news articles

Easily extendable with other models like SVM, Naive Bayes, BERT
