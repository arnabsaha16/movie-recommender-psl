import streamlit as st
import pandas as pd
import numpy as np

# Define the column names of movies to be presented to user 
movie_column_names = ['movie_id', 'title', 'genres']

# Read the movies.dat file from the URL into a dataframe
url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
movies_df = pd.read_csv(url, sep='::', header=None, names=movie_column_names, engine='python', encoding='latin1')

# Add poster image URLs from Github repository to movies_df dataframe
movies_list = movies_df['movie_id'].to_list()
poster_urls = [f"https://liangfgithub.github.io/MovieImages/{i}.jpg?raw=true" for i in movies_list]
movies_df.loc[movies_df['movie_id'].isin(movies_list), 'poster_url'] = poster_urls

# Filter the above movies dataset to get the 100 movies and corresponding titles/poster URLs chosen for display in Section 1
id_100_movies = [i for i in range(1, 201) if i % 2 == 0]
movies100_df = pd.DataFrame()
movies100_df = movies_df[movies_df['movie_id'].isin(id_100_movies)].reset_index(drop=True)

# Add 'm' as prefix to all movie_id values in movies100_df
movies100_df['movie_id'] = 'm' + movies100_df['movie_id'].astype(str)

# Function to simulate the myIBCF recommendation system
def modified_myIBCF(newuser):
    # Load the similarity matrix from Github (the same one that was generated in the last step above)
    S_file = r'https://raw.githubusercontent.com/arnabsaha16/movie-recommender-psl/refs/heads/main/similarity_matrix_100movies.csv'
    S = pd.read_csv(S_file, index_col=0, header=0)

    # Load the top 10 movies recommended by system 1 from Github
    top10movies_system1_file = r'https://github.com/arnabsaha16/movie-recommender-psl/raw/refs/heads/main/top_movies_system1.csv'
    top10movies_system1 = pd.read_csv(top10movies_system1_file, index_col=0, header=0)
    
    # Ensure newuser is a numpy array and set the index same as Similarity Matrix
    w = np.array(newuser).flatten()

    # Set user input rating values to np.nan if they are not 1, 2, 3, 4, or 5
    valid_ratings = {1, 2, 3, 4, 5}
    w = np.where(np.isin(w, list(valid_ratings)), w, np.nan)
    
    # Initialize the predictions array
    predictions = np.full(S.shape[0], np.nan)
   
    # Iterate over each movie (movie 'i' represents the movie for which predictions are being made)
    for i in range(len(predictions)):
        numerator = 0
        denominator = 0
        # Iterate over each movie 'j' which has a non-NA rating from user & shares a non-NA similarity value with movie 'i'
        for j in range(len(w)):
            if not np.isnan(w[j]) and j != i and not np.isnan(S.iloc[i, j]):
                numerator += S.iloc[i, j] * w[j]
                denominator += S.iloc[i, j]
            
        # Check for denominator being zero
        if denominator != 0:
            predictions[i] = numerator / denominator
        else:
            predictions[i] = np.nan
    
    # Convert predictions array back to a Pandas Series for easier handling and drop NA values
    predictions_series = pd.Series(predictions, index=S.index).dropna()
    
    # Get the top 10 recommendations
    top_10_recommendations = predictions_series.sort_values(ascending=False).head(10)
    ibcf_recommendations = top_10_recommendations.index.tolist()

    # Check if the number of non-NA recommendations is less than 10
    if len(ibcf_recommendations) < 10:
        # Find the shortfall
        shortfall = 10 - len(ibcf_recommendations)
        movie_list = top10movies_system1.iloc[:]['movie_id'].to_list()
        movie_list = ['m' + str(num) for num in movie_list]
        
        # Add movies from top10movies_system1 until ibcf_recommendations has 10 movie IDs 
        index = 0
        while shortfall > 0 and index < len(top10movies_system1):
            if movie_list[index] not in ibcf_recommendations:
                ibcf_recommendations.append(movie_list[index])
                shortfall -= 1
                index += 1
                
    return ibcf_recommendations

# Initialize the app
st.title('Movie Recommendations')

# Custom CSS for fixed container height 
st.markdown(""" 
	<style>
    body { 
        background-color: #f0f0f5; /* Set background color */ 
        color: #333333; /* Set text color for contrast */ 
    }
	.fixed-height {
		height: 280px;
		width: 120px;
		overflow: hidden; 
		display: flex; 
		flex-direction: column; 
		justify-content: space-between; 
	}
    .movie-row { 
        display: flex; 
        justify-content: space-between; 
        align-items: flex-start; /* Align items to the top */ 
        margin-bottom: 20px; /* Space between rows */ 
    }
    .movie-container { 
        width: 40%; /* Increase the width for each movie container */ 
        text-align: top; 
    }
    .movie-title { 
        margin-top: auto; /* Align the title */ 
        text-align: top; 
        font-weight: bold; 
    }
    .fixed-image-height { 
        height: 200px; /* Set fixed height for images */ 
        width: 150px; /* Maintain aspect ratio */ 
    }
    </style> 
	""", unsafe_allow_html=True)

# Section 1: Display the movie list
st.subheader('Movie List')
st.text("Give your ratings to one or more movies below and click on 'Recommend' button at the end to see our recommendations for you.")
cols = st.columns(5)

ratings = {}
for idx, row in movies100_df.iterrows():
    with cols[idx % 5]:
        with st.container():
            st.markdown(f"""
			    <div class="fixed-height"> 
				    <img src="{row["poster_url"]}" alt="Movie Poster" style="width: 100%;">
                    <p>{row['title']} ({row['movie_id']})</p> 
			    </div> """, unsafe_allow_html=True)
            rating = st.radio('Rate the above movie', options=["Rating not provided", 1, 2, 3, 4, 5], index=0, key=f'rating_{row["movie_id"]}')
            ratings[row["movie_id"]] = np.nan if rating == "Rating not provided" else rating

# Convert ratings to a (100,1) vector
# user_ratings_vector = np.array([ratings[movie_id] for movie_id in movies_df['movie_id']]).reshape(-1, 1)
user_ratings_vector = np.array([ratings.get(movie_id, np.nan) for movie_id in movies100_df['movie_id'].to_list()]).reshape(-1, 1)

# Recommend button
if st.button('Recommend'):
    # Get recommendations
    
    recommended_movie_ids = modified_myIBCF(user_ratings_vector)

    # Strip 'm' prefix and convert all recommended movie IDs to integers
    recommended_movie_ids_int = [int(movie_id[1:]) for movie_id in recommended_movie_ids]

    # Get the recommended movies
    recommended_movies = movies_df[movies_df['movie_id'].isin(recommended_movie_ids_int)]

    # Convert the movie_id back to 'm' prefixed strings
    recommended_movies['movie_id'] = 'm' + recommended_movies['movie_id'].astype(str)
  
    # Section 2: Display the recommended movies
    st.subheader('Recommended Movies')
    st.text('The following 10 movies are our recommendations based on your assigned ratings.')
    cols = st.columns(2)
    for idx, row in recommended_movies.iterrows():
        with cols[idx % 2]:
            with st.container():
                st.markdown(f""" 
			        <div> 
				        <img src="{row["poster_url"]}" alt="Movie Poster" style="width: 200%; height: 200%"> 
				        <p>{row['title']} ({row['movie_id']})</p> 
			        </div> """, unsafe_allow_html=True)
