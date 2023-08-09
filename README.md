# Fake-News-Detection-Model
## Project Overview
This project is dedicated to building a fake news detection model using Logistic Regression. In the modern age where misinformation can spread rapidly, having tools that can automatically detect and filter out fake news is of paramount importance. This model, though simple, serves as a baseline to tackle this problem using classic machine learning techniques.

## Disclaimer
This project was guided and inspired by @Siddhardhan's tutorial on YouTube. All credits for the original idea and methodology go to him. Any additional contributions or modifications made here are separate from the original work presented by @Siddhardhan.

## Libraries Used
pandas: For data manipulation and analysis.
numpy: For numerical operations.
re: For regular expression operations.
string: For common string operations.
nltk: Natural Language Toolkit, for text processing libraries for classification, tokenization, stemming, tagging, and parsing.
sklearn: For machine learning operations.

## Workflow
1. Data Preprocessing: This involves cleaning the data and getting it ready for the machine learning model. Any NaN values, duplicate rows, and irrelevant columns might be removed. Text data is often messy and needs to be cleaned for better results.
2. Text Cleaning and Stemming: The following steps are taken:
   - Remove punctuation and numbers.
   - Convert all text to lowercase.
   - Remove stopwords (words that are commonly used in the English language but are generally ignored in text data processing like 'and', 'the', 'is', etc.).
   - Stemming: Reduce each word to its root or base. For instance, "running" becomes "run", "runs" becomes "run".
3. Vectorization: The cleaned text data is converted to a format that can be fed into the Logistic Regression model using the TfidfVectorizer. This will convert the text data into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features.
4. Train-Test Split: The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
5. Model Training: The Logistic Regression model is trained on the training dataset. In the context of binary classification (fake news vs. true news), the model will predict a probability score between 0 and 1. Given a threshold (e.g., 0.5), if the probability score is greater than the threshold, the news article is classified as fake news (label 1). If it's less, it's classified as true news (label 0).
6. Evaluation: After the model has been trained, it's evaluated on the test set to determine its accuracy and potentially other metrics like precision, recall, and F1-score.

## How to Run
- Ensure you have all the required libraries installed.
- Load your dataset.
- Follow the code structure provided above, from importing libraries to evaluation.

## Future Work
- Experiment with different machine learning models and techniques.
- Integrate deep learning models such as RNNs or Transformers which might give better results for large texts.
- Use word embeddings like Word2Vec or GloVe for better representation of text data.
- Deploy the model as a web or mobile application for real-time fake news detection.

## Contribution
Feel free to fork this repository, make improvements, or adapt the model for your own use case. Any contributions to improve the efficiency or accuracy of the model are highly appreciated.

