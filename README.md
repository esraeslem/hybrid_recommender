# Hybrid Recommender System

A hybrid movie recommendation engine built on the [MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/) (20 million ratings, 138,493 users, 27,278 movies). It combines **user-based** and **item-based** collaborative filtering to produce a final list of recommendations for a given user.

## How it works

1. **Data prep** — merge ratings with movie titles/genres, drop movies with ≤1000 total ratings, and build a `userId x title` ratings pivot table.
2. **User-based CF** — find users who share at least 60% of the target user's watched movies, keep the ones correlated with the target user at ≥0.65, and compute a correlation-weighted average rating per movie. The top 5 movies above a 3.5 weighted score become the user-based recommendations.
3. **Item-based CF** — take the target user's most recently rated 5-star movie, correlate it against every other movie in the pivot table, and take the top 5 most correlated movies.
4. **Final output** — the 5 user-based + 5 item-based recommendations, combined into a list of 10.

## Getting started

```bash
pip install -r requirements.txt
```

Download the dataset — see [`data/README.md`](data/README.md) — and place `movie.csv` and `rating.csv` in the `data/` folder.

```bash
python hybrid_recommender.py
```

## Notes on working with a 20M-row dataset

`rating.csv` is ~690 MB / 20 million rows, which is enough to run into memory issues on modest machines if you're not careful. This implementation:

- Reads columns with compact dtypes (`int32`/`float32`) and parses the timestamp directly as `datetime`, instead of the pandas defaults (`int64`/`float64`/strings).
- Converts `title`/`genres` to the `category` dtype after merging, which collapses millions of repeated strings down to a small lookup table.
- Builds the `userId x title` pivot table via `groupby(...).mean().unstack()` (freeing intermediate DataFrames in between) rather than a single chained `pivot_table` call, which keeps peak memory noticeably lower.
- Avoids `DataFrame.corrwith()`, which internally casts the whole matrix to `float64` (a multi-GB temporary copy on this dataset); a column-by-column Pearson correlation is used instead, with an identical result.
- A duplicate-index edge case is also handled: a few different `movieId`s share the same title, which can produce duplicate `(userId, title)` pairs and duplicate index entries — both are deduplicated before pivoting/correlating (recent pandas versions raise an error on duplicate index/columns in `stack`/`unstack`).

None of this changes the output — it's the same algorithm, just able to run comfortably on machines with a few GB of free RAM rather than requiring a high-memory environment.

## Example output

For a sample user (id `108170`):

**User-based:** Mystery Science Theater 3000: The Movie (1996) · A Christmas Story (1983) · The Natural (1984) · Super Troopers (2001) · The Kid (1921)

**Item-based** (based on their most recent 5-star rating, *Wild at Heart* (1990)): My Science Project (1985) · Mediterraneo (1991) · The Old Man and the Sea (1958) · National Lampoon's Senior Trip (1995) · Clockwatchers (1997)

## License

MIT — see [LICENSE](LICENSE).
