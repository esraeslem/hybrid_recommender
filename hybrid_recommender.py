#############################################
# Hybrid Recommender System (User-Based + Item-Based)
#############################################

# For a given user, this script produces movie recommendations using both a
# user-based and an item-based collaborative filtering approach, then combines
# them into a final list of 10 recommendations (5 from each method).
#
# Dataset: MovieLens 20M (movie.csv, rating.csv) - see data/README.md for
# download instructions.

import ctypes
import gc

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

DATA_DIR = "data"

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def free_memory():
    """gc.collect() + malloc_trim: actually releases freed memory back to the
    OS (by default glibc keeps freed memory reserved inside the process, so
    RSS doesn't drop). When working with a 20M-row rating.csv on a
    memory-constrained machine, this prevents the process from crashing due
    to out-of-memory errors."""
    gc.collect()
    if _libc is not None:
        _libc.malloc_trim(0)


#############################################
# Task 1: Preparing the Data
#############################################

# Step 1: Load the movie and rating datasets.
# movieId, title and genre information for each movie
movie = pd.read_csv(
    f"{DATA_DIR}/movie.csv",
    dtype={"movieId": "int32", "title": "string", "genres": "string"},
)

# userId, movieId, rating and timestamp information
# NOTE: rating.csv has 20 million rows and is ~690 MB. Using smaller dtypes
# (int32/float32) and parsing the timestamp directly as datetime drastically
# cuts memory usage (otherwise the process can run out of memory).
rating = pd.read_csv(
    f"{DATA_DIR}/rating.csv",
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    parse_dates=["timestamp"],
)


def create_user_movie_df(movie, rating):
    # Step 2: Add the movie title and genre to the rating dataset.
    df = rating.merge(movie, how="left", on="movieId")

    # Converting title/genres to "category" shrinks the repeated text data
    # across 20 million rows from potentially tens of GB down to a few
    # hundred MB.
    df["title"] = df["title"].astype("category")
    df["genres"] = df["genres"].astype("category")

    # Step 3: Compute how many total ratings each movie received.
    # Drop movies with 1000 or fewer total ratings from the dataset.
    comment_counts = pd.DataFrame(df["title"].value_counts())
    comment_counts.columns = ["count"]
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]

    # Step 4: Build a pivot table with userId as the index, movie titles as
    # the columns, and ratings as the values.
    # NOTE: A handful of different movieIds can share the same title, which
    # occasionally causes the (userId, title) pair to repeat. pivot_table
    # aggregates duplicates with mean by default; we get the same result via
    # groupby(mean) + unstack while keeping float32, which roughly halves the
    # memory footprint (vs. float64) of the ~140k x ~3k dense matrix.
    #
    # We free `df` BEFORE the expensive unstack step; otherwise df +
    # common_movies + the unstack's temporary memory would all be alive at
    # once and could exhaust available memory.
    del df
    free_memory()

    # NOTE: groupby(...).mean() and .unstack() are deliberately split into
    # separate statements instead of one chained call. In a chained call,
    # `common_movies` would stay referenced (and in memory) until the whole
    # chain finishes, even though it's no longer needed after .mean().
    # Splitting it and deleting `common_movies` in between meaningfully
    # reduces the peak memory usage during the (expensive) unstack step.
    grouped = common_movies.groupby(["userId", "title"], observed=True)["rating"].mean()
    del common_movies
    free_memory()

    user_movie_df = grouped.unstack().astype("float32")
    del grouped
    free_memory()
    return user_movie_df


# Step 5: The steps above are wrapped into a function; now we call it.
user_movie_df = create_user_movie_df(movie, rating)
print(f"user_movie_df shape: {user_movie_df.shape}  (users x movies)")


#############################################
# Task 2: Determining the Movies Watched by the Target User
#############################################

# Step 1: The ID of the user we want to generate recommendations for (kept
# the same throughout the script so the item-based section is consistent
# with the user-based section).
random_user = 108170

# Step 2: The subset of user_movie_df belonging to the target user.
random_user_df = user_movie_df[user_movie_df.index == random_user]

# Step 3: The movies the target user has rated.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
print(f"\nUser {random_user} has rated {len(movies_watched)} of the popular movies.")


#############################################
# Task 3: Accessing the Data and IDs of Other Users Who Watched the Same Movies
#############################################

# Step 1: The columns corresponding to the movies the target user watched.
movies_watched_df = user_movie_df[movies_watched]

# Step 2: For every user, how many of the target user's watched movies they
# have also rated.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Step 3: Users who watched more than 60% of the target user's movies are
# considered "similar users".
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"].tolist()
print(f"Number of similar users (>60% overlap in watched movies): {len(users_same_movies)}")


#############################################
# Task 4: Determining the Users Most Similar to the Target User
#############################################

# Step 1: Combine the similar users with the target user.
final_df = pd.concat(
    [
        movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
        random_user_df[movies_watched],
    ]
)
# The target user has, by definition, watched 100% of their own watched
# movies, so they're automatically included in users_same_movies already -
# which creates a duplicate index entry here. Recent pandas versions don't
# allow duplicate index/columns in stack/unstack, so we clean it up.
final_df = final_df[~final_df.index.duplicated(keep="first")]

# Step 2: Correlation between users.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

# Step 3: Users whose correlation with the target user is above 0.65.
top_users = corr_df[
    (corr_df["user_id_1"] == random_user)
    & (corr_df["corr"] >= 0.65)
    & (corr_df["user_id_2"] != random_user)
][["user_id_2", "corr"]]
top_users = top_users.sort_values("corr", ascending=False).reset_index(drop=True)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
print(f"\nSimilar users with correlation >= 0.65:\n{top_users}")

# Step 4: Merge top_users with the rating dataset.
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner", on="userId")


#############################################
# Task 5: Computing the Weighted Average Recommendation Score and Keeping the Top 5 Movies
#############################################

# Step 1: weighted_rating = corr * rating
top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

# Step 2: Average weighted rating across all users, per movie.
recommendation_df = top_users_ratings.groupby("movieId")["weighted_rating"].mean().reset_index()

# Step 3: Keep movies with a weighted rating above 3.5, sorted, top 5.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values(
    "weighted_rating", ascending=False
)

# Step 4: Look up the movie titles for the recommended movies.
user_based_recommendations = movies_to_be_recommend.merge(movie, on="movieId")[
    ["movieId", "title", "weighted_rating"]
].head(5)

print("\n########## TOP 5 USER-BASED RECOMMENDATIONS ##########")
print(user_based_recommendations.to_string(index=False))


#############################################
# Item-Based Recommendation
#############################################

# Item-based recommendations are generated from the most recent movie the
# user rated 5.0 stars.
user = 108170  # (same user as above, kept consistent across the script)

# Step 1: movie and rating are already loaded above (no need to reload them).

# Step 2: movieId of the most recently rated 5.0-star movie for this user.
movie_id = (
    rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]
    .sort_values("timestamp", ascending=False)["movieId"]
    .iloc[0]
)
movie_title = movie.loc[movie["movieId"] == movie_id, "title"].iloc[0]
print(f"\nUser's most recent 5-star rating: '{movie_title}' (movieId={movie_id})")

# rating and movie are no longer needed; free them to make room for the next
# step.
del rating, movie
free_memory()

# Step 3: Filter user_movie_df down to the column for the selected movie.
movie_df = user_movie_df[movie_title]


# Step 4: Correlation between the selected movie and every other movie, sorted.
# NOTE: pandas' DataFrame.corrwith() internally casts the ENTIRE ~140k x ~3k
# matrix to float64, creating a temporary copy of roughly 3.5 GB - which can
# exhaust memory on constrained machines. Instead, we compute a
# mathematically identical Pearson correlation column-by-column, only ever
# working with small arrays at a time.
def corr_with_column(df, target_col_name):
    target = df[target_col_name].to_numpy(dtype=np.float32)
    valid_target = ~np.isnan(target)
    mat = df.to_numpy()
    corrs = np.empty(mat.shape[1], dtype=np.float64)
    for i in range(mat.shape[1]):
        col = mat[:, i]
        mask = valid_target & ~np.isnan(col)
        if mask.sum() < 2:
            corrs[i] = np.nan
            continue
        t = target[mask].astype(np.float64)
        x = col[mask].astype(np.float64)
        if t.std() == 0 or x.std() == 0:
            corrs[i] = np.nan
        else:
            corrs[i] = np.corrcoef(t, x)[0, 1]
    return pd.Series(corrs, index=df.columns)


corr_with_movie = corr_with_column(user_movie_df, movie_title).sort_values(ascending=False)

# Step 5: Top 5 movies, excluding the selected movie itself.
item_based_recommendations = corr_with_movie.drop(movie_title, errors="ignore").head(5)

print("\n########## TOP 5 ITEM-BASED RECOMMENDATIONS ##########")
print(item_based_recommendations.to_string())


#############################################
# FINAL 10 RECOMMENDATIONS (5 user-based + 5 item-based)
#############################################

print(f"\n########## FINAL 10 RECOMMENDATIONS FOR USER {random_user} ##########")
print("\n-- User-Based --")
for i, t in enumerate(user_based_recommendations["title"], 1):
    print(f"{i}. {t}")

print(f"\n-- Item-Based (based on watching/liking '{movie_title}') --")
for i, t in enumerate(item_based_recommendations.index, 1):
    print(f"{i}. {t}")
