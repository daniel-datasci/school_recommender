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
    cv_mat = count_vect.fit_transform(data)
    # Get the cosine
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat


# Recommendation System
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=5):
    # indices of the program
    program_indices = pd.Series(df.index, index=df['Programme_name']).drop_duplicates()
    # Index of the programme
    idx = program_indices[title]

    # look into the cosine matriix for that index
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_program_indices = [i[0] for i in sim_scores[1:]]
    selected_program_scores = [i[1] for i in sim_scores[1:]]

    # Get the dataframe and title
    recommended_result = df.iloc[selected_program_indices]
    recommended_result['similarity_scores'] = selected_program_scores
    final_recommendation = recommended_result[['Programme_name', 'similarity_scores', 'Programme_requirement', 'School_Name', 'University_Website', 'Specialization', 'Country']]

    return final_recommendation



# CSS STYLE
RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score:</span>{}</p>
<p style="color:blue;"><span style="color:black;">✅Requirement:</span>{}</p>
<p style="color:blue;"><span style="color:black;">👨🏻‍💻Specialization::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗Website:</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">🏫School:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🎌Country:</span>{}</p>

</div>
"""



# Search for program
def search_term_if_not_found(term, df):
	final_recommendation = df[df['Programme_name'].str.contains(term)]
	return final_recommendation



def main():

    st.title("Program Recommendation App")

    Menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", Menu)
    df = load_data("data/data.csv")

    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))

    elif choice == "Recommend":
        st.subheader("Recommend Programs")

        cosine_sim_mat = vectorize_text_to_cosine_mat(df['Programme_name'])
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number", 3, 5, 5)
        
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    # st.write(results)
                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_score = row[1][1]
                        rec_req = row[1][2]
                        rec_sch = row[1][3]
                        rec_web  = row[1][4]
                        rec_spe = row[1][5]
                        rec_coun = row[1][6]

                        #st.write("Title:", rec_title)
                        stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_req, rec_spe, rec_web, rec_sch, rec_coun), height=450)

                except:
                    results = "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    final_recommendation = search_term_if_not_found(search_term, df)
                    st.dataframe(final_recommendation)

               

    else:
        st.subheader("About")
        st.text("Built with Streamlit By Casia Growth Lab")

if __name__ == '__main__':
    main()