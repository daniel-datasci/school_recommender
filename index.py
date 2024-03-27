import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load datasets
# Load Our Dataset
def load_data(data):
    df = pd.read_csv(data)
    return df

candidate_df = load_data("data/candidate2.csv")
school_df = load_data("data/school.csv")

# Preprocess data
def preprocess_data(df):
    df.fillna('', inplace=True)
    df['text_features'] = df['Current_Job'] + ' ' + df['Qualification'] + ' ' + df['Industry']
    return df

# Preprocess data
def preprocess_data2(df):
    df.fillna('', inplace=True)
    df['text_features'] = df['Programme_name'] + ' ' + df['Programme_requirement'] + ' ' + df['Document_Requirements'] + ' ' + df['School_Name'] + ' ' + df['Country'] + ' ' + df['University_Website']
    return df

candidate_df = preprocess_data(candidate_df)
school_df = preprocess_data2(school_df)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(candidate_df['text_features'])

# Calculate cosine similarity between candidate and school data
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_vectorizer.transform(school_df['text_features']))

# Streamlit UI
def main():
    st.title('Masters Program Recommendation System - Individual Section')

    # Input fields
    name = st.text_input('Enter Your Name')
    email = st.text_input('Enter Your Email')
    current_job = st.text_input('Enter Your Current Job')
    qualification = st.text_input('Enter Your Qualification')
    years_of_experience = st.number_input('Enter Years of Experience', min_value=0)
    industry = st.text_input('Enter Your Industry')
    submit_button = st.button('Get Recommendations')

    if submit_button:
        # Prepare user profile
        user_profile = f"{current_job} {qualification} {industry}"
        user_tfidf = tfidf_vectorizer.transform([user_profile])

        # Calculate cosine similarity between user profile and programs
        user_scores = cosine_similarity(user_tfidf, tfidf_vectorizer.transform(school_df['text_features']))
        top_program_indices = user_scores.argsort()[0][-3:][::-1]

        # Display recommendations
        st.subheader(f"Hello {name},")
        st.markdown(f"We have taken a look at your profile.")
        st.markdown(f"You are a {current_job} with a total of {years_of_experience} years of experience. You are certified with {qualification}.")
        st.markdown("\nBelow are the recommendations we have for you based on your profile:\n")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        for i, idx in enumerate(top_program_indices, 1):
            program = school_df.loc[idx]
            st.markdown(f"**Recommendation {i} for {name}:**")
            st.markdown(f" Programme Name: {program['Programme_name']}")
            st.markdown(f"   Programme Requirement: {program['Programme_requirement']}")
            st.markdown(f"   Document Requirements: {program['Document_Requirements']}")
            st.markdown(f"   School Name: {program['School_Name']}")
            st.markdown(f"   Country: {program['Country']}")
            st.markdown(f"   University Website: {program['University_Website']}")
            st.markdown("")
            st.markdown("")
            st.markdown("")

if __name__ == "__main__":
    main()
