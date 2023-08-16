import streamlit as st
import gdown
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_data
def collect():
  gdown.download(id = '1GK_KpfUVfdbdJlutyaXX_Sf0uaApqXfN')#ret data
  return

@st.cache_resource
def getdata():
  gdown.download(id = '1AFL6D6TJgqpkjNAwtNrf0j0e3GadxWp_')
  gdown.download(id = '1vLnhZi7vk0Hal45b0cdJrIaPafBFbkN_')#knn model
  with open(r'/mount/src/linkedin_deployment/knn_model_linkedin.pkl', 'rb') as file:
    model = pickle.load(file)
  return model

model = getdata()

def top_4_prediction(input_data):
    df = input_data
    del df['name']
    
    #------------------------------------------------------------------------------------------------------
    for column in df.columns:
      df[column] = df[column].apply(lambda x: str(x).lower().replace(r'[^\w\s]', ''))
      df[column] = df[column].str.replace(r'fulltime|selfemployed|· self-employed|· full-time', '', flags=re.IGNORECASE).str.strip()
    #------------------------------------------------------------------------------------------------------

    titlecols = [col for col in df.columns if 'title' in col]
    df['all_titles'] = df[titlecols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)#--------------------------------------remove punctuvation
    df.drop(columns=titlecols, inplace=True)

    org_cols = ([col for col in df.columns if 'org' in col])
    df['all_org_cols'] = df[org_cols].apply(lambda row: ','.join(row.dropna().astype(str)), axis=1)
    df.drop(columns=org_cols, inplace=True)

    job_location_cols = [col for col in df.columns if 'location' in col]
    df['all_loc_cols'] = df[job_location_cols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    df.drop(columns=job_location_cols, inplace=True)

    institute_cols = [col for col in df.columns if 'institute' in col]
    df['all_instutes'] = df[institute_cols].apply(lambda row: ','.join(row.dropna().astype(str)), axis=1)
    df.drop(columns=institute_cols, inplace=True)

    degree_cols = [col for col in df.columns if 'degree' in col and 'duration' not in col]
    df['all_degree'] = df[degree_cols].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    df.drop(columns=degree_cols, inplace=True)
    
    comp_emp_count = [col for col in df.columns if 'company' in col and 'emp' in col]            # how big are their company exp's so we are totaling the info.
    df['all_comp_emp_count'] = df[comp_emp_count].sum(axis=1)
    df.drop(columns=comp_emp_count, inplace=True) #not enough data

    comp_ind = [col for col in df.columns if 'company' in col and 'industry' in col]
    df['all_comp_ind'] = df[comp_ind].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    df.drop(columns=comp_ind, inplace=True)

    #------------------------------------------------------------------------------------------------------
    def vectorize_columns_method1(df, columns_to_vectorize):
        tfidf_vectorizer_method1 = TfidfVectorizer(tokenizer=lambda x: x.split(','))
        job_titles_method1 = df[columns_to_vectorize].apply(lambda row: ','.join(row), axis=1)
        job_title_matrix_method1 = tfidf_vectorizer_method1.fit_transform(job_titles_method1)
        job_title_vector_df_method1 = pd.DataFrame(job_title_matrix_method1.toarray(), columns=tfidf_vectorizer_method1.get_feature_names_out())
        return job_title_vector_df_method1,tfidf_vectorizer_method1

    def vectorize_columns_method2(df, columns_to_vectorize):
        tfidf_vectorizer_method2 = TfidfVectorizer()
        job_titles_method2 = df[columns_to_vectorize].apply(lambda row: ' '.join(row), axis=1)
        job_title_matrix_method2 = tfidf_vectorizer_method2.fit_transform(job_titles_method2)
        job_title_vector_df_method2 = pd.DataFrame(job_title_matrix_method2.toarray(), columns=tfidf_vectorizer_method2.get_feature_names_out())
        return job_title_vector_df_method2,tfidf_vectorizer_method2

    columns_to_vectorize_1 = ['all_org_cols','all_instutes']
    columns_to_vectorize_2 = [ 'all_titles', 'all_loc_cols', 'all_degree', 'all_comp_ind']

    vectorized_df_method1,tfidf_vectorizer_method1 = vectorize_columns_method1(df, columns_to_vectorize_1)
    vectorized_df_method2,tfidf_vectorizer_method2 = vectorize_columns_method2(df, columns_to_vectorize_2)

    # Concatenate the vectorized DataFrames horizontally
    result_df = pd.concat([df, vectorized_df_method1, vectorized_df_method2], axis=1)

    # Drop the non-vectorized columns
    result_df = result_df.drop(columns=columns_to_vectorize_1)
    result_df = result_df.drop(columns=columns_to_vectorize_2)
    #------------------------------------------------------------------------------------------------------
    colsdf = pd.read_csv(r'/mount/src/linkedin_deployment/x_train.csv')
    colsdf = pd.DataFrame(columns=colsdf.columns.to_list())

    for column in colsdf.columns:
        if column in result_df.columns:
            colsdf[column] = result_df[column]
    colsdf = colsdf.fillna(0)
    #------------------------------------------------------------------------------------------------------
    orgdf = pd.read_csv(r'/mount/src/linkedin_deployment/retrivaldata.csv')

    user_input = colsdf.iloc[0]  # Change the index (5) to any user index you want to test
    user_input = user_input.values.reshape(1, -1)  # Reshape to 2D array

    distances, indices = model.kneighbors(user_input, n_neighbors=4)
    
    top_4_indices = indices[0]

    top_4_profiles = orgdf.iloc[top_4_indices]
    
    top_4_profiles = top_4_profiles.dropna(axis=1, how='all')
    top_4_profiles = top_4_profiles.loc[:, (top_4_profiles != 0.0).any()]
    top_4_profiles = top_4_profiles.reset_index(drop=True)
    
    cosine_similarity_scores = [1 - distance for distance in distances]
    st.write( f'KNN Distances from profiles', distances)

    return(top_4_profiles)

def main():
    a = collect()
    st.write('Welcome to the Application, *Find Matching Founder Profiles* :sunglasses:')

    st.title('Enter your profile Details')
    
    main = {}
    # Basic Information
    name = st.text_input("Name")
    location = st.text_input("Location")
    
    main.update({
      'name':name,
      'location':location
    })

    # Previous Jobs
    num_previous_jobs = st.number_input("Number of Previous Jobs", min_value=0, max_value=15, value=0)

    for i in range(num_previous_jobs):
        st.subheader(f"Previous Job {i+1}")
        org = st.text_input(f"Organization {i+1}")
        title = st.text_input(f"Title {i+1}")
        duration = st.number_input(f"Duration {i + 1} in Months", min_value=0, max_value=999,value=0)
        job_location = st.text_input(f"Job Location {i+1}")

        main[f"org_{i}"] = org
        main[f"title_{i}"] = title
        main[f"job_{i}_duration"] = duration
        main[f"job_{i}_location"] = job_location
    
    # Education
    num_education_entries = st.number_input("Number of Education Entries", min_value=0, max_value=3, value=0)

    for i in range(num_education_entries):
        st.subheader(f"Education Entry {i+1}")
        institute = st.text_input(f"Institute {i+1}")
        degree = st.text_input(f"Degree {i+1}")
        degree_duration = st.number_input(f"Degree Duration {i+1} in Months", min_value=0, max_value=999, value=0)
        
        main[f"institute_{i}"] = institute
        main[f"degree_{i}"] = degree
        main[f"degree_{i}_duration"] = degree_duration

    # Company Information
    for i in range(num_previous_jobs):
        st.subheader(f"Company {i + 1}")
        emp_count = st.number_input(f"Employee Count {i + 1}", min_value=0)
        industry = st.text_input(f"Industry {i + 1}")
        
        main[f"company_{i}_emp_count"] = emp_count
        main[f"company_{i}_industry"] = industry

    user_input = pd.DataFrame(main, index=[0])

    # prediction

    top_4_profiles = pd.DataFrame()
    
    # button
    if st.button('Check'):
        top_4_profiles = top_4_prediction(user_input)
        
    st.write(top_4_profiles)
    

if __name__ == '__main__':
    main()
