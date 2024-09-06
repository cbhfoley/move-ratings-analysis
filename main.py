import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

# Configure page
st.set_page_config(page_title="MovieLens Data Analysis", layout="centered")

st.markdown("""
# MovieLens Data Analysis Dashboard
This interactive dashboard allows you to analyze and visualize data from the MovieLens dataset. You can explore movie ratings, apply genre filters, and even train a machine learning model to predict movie ratings.
""")

# File upload functionality
st.markdown("""
### Step 1: Upload MovieLens Data
Please upload your ratings and movies CSV files to begin the analysis.
""")
uploaded_ratings = st.file_uploader("Upload your ratings CSV file", type=['csv'], key="ratings")
uploaded_movies = st.file_uploader("Upload your movies CSV file", type=['csv'], key="movies")

# Load and display the datasets if uploaded
if uploaded_ratings is not None and uploaded_movies is not None:
    ratings = pd.read_csv(uploaded_ratings)
    movies = pd.read_csv(uploaded_movies)

    st.markdown("""
    ### Data Preview
    Below is a sample of the data you've uploaded:
    """)
    st.write("Ratings Data Sample:")
    st.dataframe(ratings.head())
    st.write("Movies Data Sample:")
    st.dataframe(movies.head())

    # One-hot encoding for genres
    movies = movies.join(movies['genres'].str.get_dummies('|'))

    # Calculate average rating and number of ratings for each movie
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']

    # Merge these features back into the movies DataFrame
    movies = pd.merge(movies, movie_stats, on='movieId', how='left')

    # Merge the ratings and movies DataFrames on the 'movieId' column
    merged_df = pd.merge(ratings, movies, on='movieId')

    # Calculate user-specific features
    user_stats = ratings.groupby('userId')['rating'].agg(['mean', 'count']).reset_index()
    user_stats.columns = ['userId', 'user_avg_rating', 'user_num_ratings']

    # Merge these user-specific features
    merged_df = pd.merge(merged_df, user_stats, on='userId', how='left')

    # Filtering options section
    st.markdown("""
    ### Step 2: Filter by Genres
    Use the options below to filter the dataset by specific genres. This filter will affect the visualizations and model training.
    """)
    genre_options = movies['genres'].str.get_dummies('|').columns.tolist()
    selected_genres = st.multiselect('Select Genres', options=genre_options, default=genre_options)


    # Function to filter movies based on the exact genre match
    def filter_by_genres(row, selected_genres):
        movie_genres = set(row['genres'].split('|'))  # Split genres into a set
        return movie_genres.issubset(selected_genres)  # Check if all movie genres are in selected genres


    # Apply filters
    if selected_genres:
        merged_df = merged_df[merged_df.apply(filter_by_genres, axis=1, selected_genres=selected_genres)]

    # Visualizations section with expander
    st.markdown("""
    ### Step 3: Visualize the Data
    Expand the section below to explore different visualizations of the data, including rating distributions, genre distributions, and average ratings by genre.
    """)

    # Flag to track whether the visualization expander has been expanded
    has_expanded_visualizations = False

    with st.expander("Show/Hide Visualizations"):
        has_expanded_visualizations = True  # Set flag to True when user expands visualizations
        if not merged_df.empty:
            # Plot the distribution of ratings
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(merged_df['rating'], bins=10, kde=False, ax=ax)
            ax.set(title='Distribution of Movie Ratings', xlabel='Rating', ylabel='Count')
            st.pyplot(fig)

            # Plot the number of ratings per movie
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(merged_df['num_ratings'], bins=20, kde=False, ax=ax)
            ax.set(title='Number of Ratings per Movie', xlabel='Number of Ratings', ylabel='Count')
            st.pyplot(fig)

            # Define broad genre categories
            genre_mapping = {
                'Action': ['Action', 'Adventure', 'Sci-Fi', 'IMAX'],
                'Drama': ['Drama', 'Romance', 'Mystery', 'Film-Noir'],
                'Comedy': ['Comedy', 'Children', 'Animation', 'Musical'],
                'Documentary': ['Documentary'],
                'Horror': ['Horror', 'Thriller'],
                'Crime': ['Crime'],
                'Fantasy': ['Fantasy'],
                'Western': ['Western'],
                'War': ['War'],
            }

            # Apply broad genre mapping
            merged_df['broad_genre'] = merged_df['genres'].apply(lambda genres: next(
                (broad_genre for broad_genre, sub_genres in genre_mapping.items() if
                 any(sub_genre in genres for sub_genre in sub_genres)),
                'Other'
            ))

            # Calculate average ratings by broad genre
            avg_ratings_by_broad_genre = merged_df.groupby('broad_genre')['rating'].mean().sort_values(ascending=False)

            # Plot average ratings by broad genre
            if not avg_ratings_by_broad_genre.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x=avg_ratings_by_broad_genre.index, y=avg_ratings_by_broad_genre.values)
                ax.set(title='Average Ratings by Broad Genre', xlabel='Genre', ylabel='Average Rating')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.write("No data available for the selected broad genres to plot the bar chart.")

            # Calculate genre distribution
            genre_counts = merged_df['genres'].str.get_dummies(sep='|').sum().sort_values(ascending=False)

            # Ensure there are enough genres to plot
            if not genre_counts.empty:
                top_n = 15
                top_genres = genre_counts[:top_n]
                other_genres = genre_counts[top_n:].sum()
                genre_counts_to_plot = pd.concat([top_genres, pd.Series({'Others': other_genres})])

                # Plot the pie chart
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.tab20c.colors
                patches, texts, autotexts = plt.pie(
                    genre_counts_to_plot,
                    startangle=90,
                    autopct='%1.1f%%',
                    colors=colors[:len(genre_counts_to_plot)],
                    pctdistance=1.1,
                    counterclock=False
                )

                # Dynamically adjust pie chart title based on genres selected
                pie_chart_title = "Genre Distribution in Movies" if len(selected_genres) == len(
                    genre_options) else f"Genre Distribution for Selected Genres"
                plt.legend(patches, genre_counts_to_plot.index, loc='best', bbox_to_anchor=(1, 0.5))
                plt.title(pie_chart_title)
                plt.axis('equal')
                st.pyplot(fig)
            else:
                st.write("No data available for the selected genres to plot the pie chart.")
        else:
            st.write("No data available after filtering to display visualizations.")

    # Only show the model training section if the visualizations have been expanded
    if has_expanded_visualizations:
        st.markdown("""
        ### Step 4: Train the Prediction Model
        Once you've explored the data and applied any genre filters, click the button below to train a linear regression model that predicts movie ratings based on user and movie features.
        """)
        if st.button('Train Model'):
            # Select the new features for prediction
            X = merged_df[['avg_rating', 'num_ratings', 'user_avg_rating', 'user_num_ratings'] + list(
                movies.columns.difference(['movieId', 'title', 'genres']))]
            y = merged_df['rating']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict the ratings for the test set
            y_pred = model.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Mean Absolute Error: {mae}")

            # Check if a genre filter is applied
            filter_note = ""
            if len(selected_genres) < len(genre_options):
                filter_note = " (Filtered by Genres)"

            # Plot Actual vs. Predicted Ratings
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_pred)
            ax.set_xlabel('Actual Ratings')
            ax.set_ylabel('Predicted Ratings')
            ax.set_title(f'Actual vs. Predicted Ratings (Linear Regression{filter_note})')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--',
                     linewidth=2)
            st.pyplot(fig)
