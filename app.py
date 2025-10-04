import streamlit as st
import pandas as pd
import pickle
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load dataset, model, and vectorizer
@st.cache_data
def load_data_and_model():
    # Load dataset
    dataset_path = 'drugsComTrain_raw.tsv'  # Replace with the actual dataset path
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Load trained model
    model_path = 'model.pkl'  # Replace with the actual model file path
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load TF-IDF vectorizer
    vectorizer_path = 'vectorizer.pkl'  # Replace with the actual vectorizer file path
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    return model, vectorizer ,df

# Load the data and model
model, vectorizer ,df= load_data_and_model()

# Function to extract top drugs
def top_drugs_extractor(pred,df):
    ## Function for Extracting Top drugs
    
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==pred]['drugName'].head(3).tolist()
    

    return drug_lst
stop = set(stopwords.words('english'))  # Use a set for faster lookup
lemmatizer = WordNetLemmatizer()
def review_to_words(raw_review):
        """
        Cleans raw review text by removing HTML, non-alphabetic characters, stopwords,
        and lemmatizing the words.
        """
        # 1. Remove HTML tags
        review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
        # 2. Remove non-alphabetic characters
        letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
        # 3. Convert to lowercase and split into words
        words = letters_only.lower().split()
        # 4. Remove stopwords
        meaningful_words = [w for w in words if w not in stop]
        # 5. Lemmatize words
        lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
        # 6. Join words into a single string
        return ' '.join(lemmatized_words)


# Streamlit UI
st.title("Drug Recommendation System")

st.header("Enter Health Condition")
user_input = st.text_input("Describe your health problem")
df_test = pd.DataFrame({'test_sent':[user_input]})
df_test["test_sent"] = df_test["test_sent"].apply(review_to_words)
tfidf_bigram = vectorizer.transform(df_test["test_sent"])
pred = model.predict(tfidf_bigram)
a = top_drugs_extractor(pred[0],df)
if st.button("Get Top Drugs"):
    if user_input:
        try:
            top_drugs = a
            condition = pred[0]
            st.success(f"Top Recommended Drugs for {condition}:")
            #st.write(condition)
            for i, drug in enumerate(top_drugs, 1):
                st.write(f"{i}. {drug}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a health condition.")