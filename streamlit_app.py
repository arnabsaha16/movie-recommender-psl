import streamlit as st
import pandas as pd
import numpy as np

# Define the column names of movies to be presented to user 
movie_column_names = ['movie_id', 'title', 'genres']

# Read the movies.dat file from the URL into a dataframe
url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
movies_df = pd.read_csv(url, sep='::', header=None, names=movie_column_names, engine='python', encoding='latin1')

# Filter the above movies dataset to get the 100 movies and corresponding titles chosen for the app
id_100_movies = [i for i in range(1, 201) if i % 2 == 0]
movies100_df = pd.DataFrame()
movies100_df = movies_df[movies_df['movie_id'].isin(id_100_movies)].reset_index(drop=True)

# Add poster image URLs from Github repository to movies100_df dataframe
poster_urls = [f"https://github.com/arnabsaha16/movie-recommender-psl/blob/main/{i}.jpg?raw=true" for i in range(1, 201) if i % 2 == 0]
movies100_df.loc[movies100_df['movie_id'].isin(id_100_movies), 'poster_url'] = poster_urls

# Add 'm' as prefix to all movie_id values in movies100_df
movies100_df['movie_id'] = 'm' + movies100_df['movie_id'].astype(str)

# Function to simulate the myIBCF recommendation system
def myIBCF(user_ratings):
    # Sample recommendation logic (Replace with your actual logic)
    recommended_movie_ids = ['m2', 'm4', 'm6', 'm8', 'm10', 'm12', 'm14', 'm16', 'm18', 'm20']  # Example output
    return recommended_movie_ids

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
        width: 30%; /* Increase the width for each movie container */ 
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
            #st.image(row["poster_url"], caption=f"{row['title']} ({row['movie_id']})", use_container_width=True)
            rating = st.radio('Rate the above movie', options=["Rating not provided", 1, 2, 3, 4, 5], index=0, key=f'rating_{row["movie_id"]}')
            ratings[row["movie_id"]] = np.nan if rating == "Rating not provided" else rating

# Convert ratings to a (100,1) vector
# user_ratings_vector = np.array([ratings[movie_id] for movie_id in movies_df['movie_id']]).reshape(-1, 1)
user_ratings_vector = np.array([ratings.get(movie_id, np.nan) for movie_id in movies100_df['movie_id'].to_list()]).reshape(-1, 1)

# Recommend button
if st.button('Recommend'):
    # Get recommendations
    recommended_movie_ids = myIBCF(user_ratings_vector)
    
    # Get the recommended movies
    recommended_movies = movies100_df[movies100_df['movie_id'].isin(recommended_movie_ids)]
    
    # Section 2: Display the recommended movies
    st.subheader('Recommended Movies')
    cols = st.columns(5)
    for idx, row in recommended_movies.iterrows():
        with cols[idx % 5]:
            #st.image(row["poster_url"], caption=f"{row['title']} ({row['movie_id']})", use_container_width=True)
            with st.container():
                st.markdown(f""" 
			        <div class="fixed-height"> 
				        <img src="{row["poster_url"]}" alt="Movie Poster" style="width: 100%;"> 
				        <p>{row['title']} ({row['movie_id']})</p> 
			        </div> """, unsafe_allow_html=True)
