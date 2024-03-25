# core pkg
import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets

#Load Our Dataset
def load_data(data):
    df = pd.read_csv(data)
    return df


candidates_df = load_data("data/candidate.csv")
schools_df = load_data("data/school.csv")
requirements_df = load_data("data/requirements.csv")



# Merge requirements with schools on Programme_name
schools_requirements_df = pd.merge(schools_df, requirements_df, on="Programme_name")

# Convert candidate and program requirements to text for vectorization
candidates_df["candidate_text"] = candidates_df["Name"] + " " + candidates_df["Job"]
schools_requirements_df["program_text"] = schools_requirements_df["Programme_name"] + " " + schools_requirements_df["Programme_requirement"]

# Fill missing values in candidate_text column with empty strings
candidates_df["candidate_text"].fillna("", inplace=True)

# Vectorize candidate and program texts
vectorizer = TfidfVectorizer(stop_words="english")
candidate_matrix = vectorizer.fit_transform(candidates_df["candidate_text"])
program_matrix = vectorizer.transform(schools_requirements_df["program_text"])

# Calculate cosine similarity between candidate and program matrices
cosine_similarities = linear_kernel(candidate_matrix, program_matrix)

# Function to recommend programs for a candidate


# Function to recommend programs for a candidate
def recommend_programs(candidate_id):
    # Find index of candidate
    candidate_index = candidates_df.index[candidates_df["ID"] == candidate_id].tolist()
    
    if not candidate_index:
        return []  # Return empty list if candidate ID not found
    
    candidate_index = candidate_index[0]

    # Calculate similarities for the candidate
    candidate_similarities = cosine_similarities[candidate_index]

    # Get top 5 most similar programs
    top_program_indices = candidate_similarities.argsort()[-5:][::-1]

    # Display recommended programs
    recommended_programs = []
    for idx in top_program_indices:
        program_info = schools_requirements_df.loc[idx, ["Programme_name", "School_Name", "Specialization", "Country", "University_Website", "Programme_requirement", "Document_Requirements"]]
        recommended_programs.append(program_info)

    return recommended_programs

# Function to display candidate information in a table
def display_candidate_info(candidate_info):
    candidate_table = pd.DataFrame(candidate_info, columns=["Name", "Job", "LinkedIn URL"])
    st.write(candidate_table)




# Streamlit web application
st.set_page_config(page_title="Master's Program Recommendation", page_icon=":mortar_board:")

# Header
st.title("Master's Program Recommendation System")
st.write("Welcome to the Master's Program Recommendation System! Enter your candidate ID to get personalized program recommendations.")

# Candidate ID input
candidate_id = st.text_input("Enter your candidate ID:")

# Recommendation button
if st.button("Get Recommendations"):
    # Validate candidate ID
    if not candidate_id.isdigit() or int(candidate_id) not in candidates_df["ID"].values:
        st.error("Please enter a valid candidate ID.")
    else:
        # Get recommendations for the candidate
        recommended_programs = recommend_programs(int(candidate_id))

        # Display candidate information
        candidate_info = candidates_df.loc[candidates_df["ID"] == int(candidate_id), ["Name", "Job", "url"]]
        candidate_info.columns = ["Name", "Job", "LinkedIn URL"]  # Update column name for LinkedIn URL
        st.subheader("Candidate Information")
        display_candidate_info(candidate_info)

        # Display recommended programs if available
        if recommended_programs:
            st.subheader("Recommended Programs")
            for program in recommended_programs:
                st.write(program)
        else:
            st.write("No programs recommended for this candidate.")

