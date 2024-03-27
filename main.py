# core pkg
import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# Set page width
st.set_page_config(layout="wide")


# Load datasets

# Load Our Dataset
def load_data(data):
    df = pd.read_csv(data)
    return df


candidate_df = load_data("data/candidate.csv")
school_df = load_data("data/school.csv")
requirements_df = load_data("data/requirements.csv")


# Preprocess data
candidate_df.fillna('', inplace=True)  # Fill missing values with empty string
school_df.fillna('', inplace=True)
requirements_df.fillna('', inplace=True)

# Combine text features for candidate and school data
candidate_df['text_features'] = candidate_df['Name'] + ' || ' + candidate_df['Email'] + ' || ' + \
                                candidate_df['Current_Job'] + ' || ' + candidate_df['Qualification']

school_df['text_features'] = school_df['Programme_name'] + ' || ' + school_df['Programme_requirement'] + ' || ' + school_df['Document_Requirements'] + ' || ' + \
                              school_df['School_Name'] + ' || ' + school_df['Country'] + ' || ' + school_df['University_Website']

# Combine all text features for TF-IDF fitting
all_text_features = pd.concat([candidate_df['text_features'], school_df['text_features']])

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(all_text_features)

# Split TF-IDF matrix into candidate and school parts
candidate_tfidf_matrix = tfidf_matrix[:len(candidate_df)]
school_tfidf_matrix = tfidf_matrix[len(candidate_df):]

# Calculate cosine similarity between candidate and school data
cosine_sim = cosine_similarity(candidate_tfidf_matrix, school_tfidf_matrix)

# Function to recommend programs for a candidate
def recommend_program(candidate_id):
    # Get top recommended program for the candidate
    candidate_scores = cosine_sim[candidate_id]
    top_program_index = candidate_scores.argmax()  # Index of top recommended program
    top_program = school_df.iloc[top_program_index]

    # Prepare output DataFrame
    output_df = candidate_df.loc[candidate_id, ['Name', 'Email', 'Current_Job', 'Qualification']]
    output_df = pd.concat([output_df, top_program[['Programme_name', 'Programme_requirement', 'Document_Requirements', 'School_Name', 'Country', 'University_Website']]], axis=0)

    return output_df

# Streamlit UI
def main():
    st.title('Masters Program Recommendation System')

    # Define the button label
    button_label = 'Switch To Recommend Individual Profile [Click Here](https://example.com)'

    # Create the button with Markdown
    if st.button(button_label):
        # Add your functionality here
        st.write("Button clicked!")

    # Upload candidate.csv file
    candidate_file = st.file_uploader('Upload candidate.csv file', type='csv')

    if candidate_file is not None:
        # Read candidate data from uploaded file
        candidate_data = pd.read_csv(candidate_file)

        # Preprocess candidate data
        candidate_data.fillna('', inplace=True)
        candidate_data['text_features'] = candidate_data['Name'] + ' || ' + candidate_data['Email'] + ' || ' + \
                                          candidate_data['Current_Job'] + ' || ' + candidate_data['Qualification']

        # Transform candidate text data using pre-fit TF-IDF vectorizer
        candidate_tfidf_matrix = tfidf_vectorizer.transform(candidate_data['text_features'])

        # Calculate cosine similarity between candidate and school data
        candidate_scores = cosine_similarity(candidate_tfidf_matrix, school_tfidf_matrix)

        # Get top recommended program for each candidate
        recommended_programs = []
        for i in range(len(candidate_data)):
            recommended_program = recommend_program(i)
            recommended_programs.append(recommended_program)

        # Concatenate recommended programs for all candidates
        final_recommendations = pd.concat(recommended_programs, axis=1).T.reset_index(drop=True)

        # Display recommended programs
        st.write(final_recommendations)

if __name__ == "__main__":
    main()
