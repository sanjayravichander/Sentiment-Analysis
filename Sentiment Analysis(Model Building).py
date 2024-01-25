# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 06:45:14 2024

@author: DELL
"""
# Sentiment Analysis

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Importing the Data
df = pd.read_csv("C:\\Users\\DELL\\Downloads\\Final Projects\\healthcare_reviews.csv")
## Adding Index column
df=df.reset_index()
df['Review_Text'].fillna('NIL', inplace=True)
text=df['Review_Text']

## Text Prepreocessing
#Converting to lower
text=text.str.lower()

##Remove spl char and numbers
text = text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

## Tokenization
text=text.apply(word_tokenize)

## Removing stop words
stop_words=set(stopwords.words('english'))
text = text.apply(lambda x: [word for word in x if word not in stop_words])

# Stemming
#porter = PorterStemmer()
#text = text.apply(lambda x: [porter.stem(word) for word in x])

lemmatizer = WordNetLemmatizer()
text = text.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Convert back to text
text = text.apply(lambda x: ' '.join(x))

## Converting series to DataFrame
text = pd.DataFrame({'Preprocessed_Text': text})

#Concatinating the preprocessed text column along with df
df = pd.concat([df, text], axis=1)

## Dropping the non preprocessed ['Review_Text'] column
df=df.drop(columns=['Review_Text'])


## Implementing pretrained model from BERT which is done by GOOGLE on twitter sentiment Analysis Data.
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm


## Importing the pretrained model which has stored weights in it which will acts as Learning Rate and also contains all the preprocessing steps involved
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Defining a function for the model
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

# Storing the scores of corresponding text and storing it in result dict
result = {}
for i, rows in tqdm(df.iterrows(), total=len(df), desc="Calculating Sentiments"):
    # Check if 'Review_Text' is not NaN
    if not pd.isna(rows['Preprocessed_Text']):
        text = rows['Preprocessed_Text']
        key = f"{i}"
        result[key]=polarity_scores_roberta(text)

# Transposing the resut series as polarity scores for every sentence arranged in columns and converting that to DataFrame
result_df=pd.DataFrame(result).T

# Adding Index column
result_df=result_df.reset_index()

# Merginf the result_df DataFrame which contains results and df DataFrame and naming this DataFrame as result_df_1
result_df_1 = pd.merge(result_df, df, how='left', left_index=True, right_index=True)


# Assuming 'roberta_pos', 'roberta_neg', 'roberta_neu' are the columns with probability scores and Providing a Seperate column which will tell po,neu,neg
result_df_1['Sentiment_Label'] = result_df_1[['roberta_pos', 'roberta_neg', 'roberta_neu']].idxmax(axis=1).apply(lambda x: x.split('_')[-1])


# Mapping values in  Sentiment Label to positive,neutral,negative
sentiment_mapping = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}

## Implimenting the Mapping
result_df_1['Sentiment_Label'] = result_df_1['Sentiment_Label'].map(sentiment_mapping)

#Dropping the Indexes
result_df_1=result_df_1.drop(columns=['index_x','index_y'])
result_df_2=result_df_1[['Preprocessed_Text','Rating','Sentiment_Label']]

## A plot for seeing Frequency of those sentiments occured throughout the Dataset
import seaborn as sns

# Distribution of Sentiment Labels
sns.countplot(x='Sentiment_Label', data=result_df_2)
plt.title('Distribution of Sentiment Labels')
plt.show()

# Distribution of Ratings
sns.countplot(x='Rating', data=result_df_2)
plt.title('Distribution of Ratings')
plt.show()

# Sentiment distribution across different ratings
sns.countplot(x='Rating', hue='Sentiment_Label', data=result_df_2)
plt.title('Sentiment Distribution Across Ratings')
plt.show()

## Taking required Columns for Model Building
final_df=result_df_2[['Preprocessed_Text','Sentiment_Label']]
final_df['Sentiment_Label']=final_df['Sentiment_Label'].map({'positive':1,'neutral':0,'negative':-1}).astype('int')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Converting the Text column to Vectors which is a king of Sparse matrix where distinct values given to each words in sentence
vec = CountVectorizer(binary=True)
train_data = vec.fit_transform(final_df['Preprocessed_Text'])
train_labels = final_df['Sentiment_Label']


# Train the logistic regression classifier
classifier = LogisticRegression()
classifier.fit(train_data, train_labels)

# Evaluate the model on the training set
train_predictions = classifier.predict(train_data)

# Calculate confusion matrix and accuracy
c = confusion_matrix(train_labels, train_predictions)
a = accuracy_score(train_labels, train_predictions)
a
c
# Making predictions after training the model to check if it is working properly
#prediction = classifier.predict(vec.transform(["mixed feeling experience"]))
#prediction_1 = classifier.predict(vec.transform(["bad experience healthcare provider avoid possible"]))
#prediction_2 = classifier.predict(vec.transform(["healthcare provider excellent great experience"]))

# Print results
#print("Confusion Matrix:\n", c)
#print("Accuracy:", a)

# Print predictions
#print("Prediction:",prediction)
#print("Prediction 1:", prediction_1)
#print("Prediction 2:", prediction_2)

# Importing pickle to save the model
import pickle
file="C:\\Users\\DELL\\Downloads\\Sentiment Analysis\\Log_Model_S.pkl"
with open(file,'wb') as f:
    pickle.dump(classifier,f)
with open(file,'rb') as f:
    pickle.load(f)
file_vec="C:\\Users\\DELL\\Downloads\\Sentiment Analysis\\vec.pkl"
with open(file_vec,'wb') as g:
    pickle.dump(vec,g)
with open(file_vec,'rb') as g:
    pickle.load(g)
    
