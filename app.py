# core pkg
import streamlit as st
import streamlit.components.v1 as stc


# load EDA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

#Load Our Dataset
def load_data(data):
    df = pd.read_csv(data)
    return df


# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit.transform(data)
    # Get the cosine
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat


# Recommendation System
def 

def main():

    st.title("Program Recommendation App")

    Menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", Menu)
    df = load_data("data/data.csv")
    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))

    elif choice == "Recommend":
        st.subheader("Recommend Courses")

        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number", 3, 5, 5)
        
        if st.button("Recommend"):
            if search_term is not None:
                pass

    else:
        st.subheader("About")
        st.text("Built with Streamlit By Casia Growth Lab")

if __name__ == '__main__':
    main()