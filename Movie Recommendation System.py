import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data from CSV file
MovieData = pd.read_csv("movies.csv")

# Show all rows and columns for better exploration (optional)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Select relevant movie features for recommendation
MovieData = MovieData.iloc[:, [2, 5, 7, 21, 23]]

# Check for missing values
print(MovieData.isnull().sum())

# Fill missing values with empty string (alternative approaches exist)
for col in MovieData:
    MovieData[col] = MovieData[col].fillna("")

# Combine relevant features into a single string for each movie
combine_data = MovieData.iloc[:, 2] + MovieData.iloc[:, 5] + MovieData.iloc[:, 7] + MovieData.iloc[:, 21] + MovieData.iloc[:, 23]

# Replace a specific string with a sample (assuming it's an outlier)
combine_data.replace(
    "NaN",
    "Action Adventure Fantasy,corruption elves dwarves orcs middle-earth (to...),The Hobbit: The Battle of the Five Armies,Martin Freeman Ian McKellen Richard Armitage K...,Peter Jackson",
    inplace=True,
)

# Get summary statistics of the combined data (optional)
combine_data.describe()

# Get information about the DataFrame (columns, data types etc.)
MovieData.info()

# Create TF-IDF vectorizer to represent text data numerically
feature = TfidfVectorizer()

# Fit the vectorizer on the combined movie descriptions
extract = feature.fit_transform(combine_data)

# Calculate cosine similarity between all movies
similarity = cosine_similarity(extract)

# Function to get movie recommendations based on user input
def MovieSuggestion():
  movieName = input("Enter Movie Name :")

  # Get a list of all movie titles from the DataFrame
  titleList = MovieData["original_title"].tolist()

  # Find the closest match to the user's input using fuzzy matching
  closeMatch = difflib.get_close_matches(movieName, titleList)[0]

  # Get the index of the closest matching movie
  Movie_index = MovieData[MovieData.original_title == closeMatch]['index'].values[0]

  # Get a list of similarity scores for all movies with the target movie
  similarityScore = list(enumerate(similarity[Movie_index]))

  # Sort movies by their similarity score in descending order
  sortMovies = sorted(similarityScore, key=lambda x: x[1], reverse=True)

  print("Movies Suggestion to you:")
  i = 1
  for movies in sortMovies:
    index = movies[0]
    title = MovieData[MovieData.index == index]['original_title'].values[0]
    # Recommend only the top 10 most similar movies
    if i < 11:
      print(i, ".", title)
    i += 1

# Run the movie suggestion function
MovieSuggestion()
