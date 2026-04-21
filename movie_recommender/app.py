from flask import Flask, jsonify, render_template, request

import hybrid_recommender

app = Flask(__name__, template_folder='templates')
hybrid_recommender.warmup_model()


def _genre_options() -> list[str]:
    options = hybrid_recommender.get_available_genres()
    if options:
        return options
    # Defensive fallback for rare stale-process situations.
    return [
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Family",
        "Fantasy",
        "History",
        "Horror",
        "Music",
        "Mystery",
        "Romance",
        "Science Fiction",
        "Thriller",
        "TV Movie",
        "War",
        "Western",
    ]


def _parse_optional_year(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    if not value.isdigit():
        return None
    return int(value)


def _parse_optional_float(value: str) -> float | None:
    value = (value or "").strip().replace(",", ".")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@app.route("/")
def main():
    genre_options = _genre_options()
    return render_template(
        "main_page.html",
        year_mode="between",
        year_from="",
        year_to="",
        min_vote_average="",
        sort_by="hybrid",
        selected_genres=genre_options,
        genre_options=genre_options,
    )

@app.route("/", methods=['GET','POST'])
def results():
    if request.method == 'POST':
        genre_options = _genre_options()
        my_favorite = request.form.get('my_favorite', '').strip()
        year_from_raw = request.form.get('year_from', '')
        year_to_raw = request.form.get('year_to', '')
        year_mode = request.form.get('year_mode', 'between')
        min_vote_average_raw = request.form.get('min_vote_average', '')
        sort_by = request.form.get('sort_by', 'hybrid')
        selected_genres = request.form.getlist('genres')
        year_from = _parse_optional_year(year_from_raw)
        year_to = _parse_optional_year(year_to_raw)
        min_vote_average = _parse_optional_float(min_vote_average_raw)

        if year_mode == "after":
            year_to = None
        elif year_mode == "before":
            year_from = None
        elif year_mode == "all":
            year_from = None
            year_to = None

        movie_name, recommendations = hybrid_recommender.recommend_movies(
            my_favorite,
            year_from=year_from,
            year_to=year_to,
            min_vote_average=min_vote_average,
            sort_by=sort_by,
            genres=selected_genres,
        )
        if movie_name is None:
            return render_template(
                "recommends.html",
                movie_name=my_favorite,
                recommendations=[],
                error_message="Film bulunamadı. Lütfen farklı bir ad deneyin.",
                year_from=year_from_raw,
                year_to=year_to_raw,
                year_mode=year_mode,
                min_vote_average=min_vote_average_raw,
                sort_by=sort_by,
                selected_genres=selected_genres,
                genre_options=genre_options,
            )
        return render_template(
            "recommends.html",
            movie_name=movie_name,
            recommendations=recommendations,
            error_message=None,
            year_from=year_from_raw,
            year_to=year_to_raw,
            year_mode=year_mode,
            min_vote_average=min_vote_average_raw,
            sort_by=sort_by,
            selected_genres=selected_genres,
            genre_options=genre_options,
        )
    genre_options = _genre_options()
    return render_template(
        "recommends.html",
        movie_name="",
        recommendations=[],
        error_message="Bir sorun oluştu!",
        year_from="",
        year_to="",
        year_mode="between",
        min_vote_average="",
        sort_by="hybrid",
        selected_genres=genre_options,
        genre_options=genre_options,
    )


@app.route("/api/suggest")
def suggest_movies():
    query = request.args.get("q", "").strip()
    suggestions = hybrid_recommender.suggest_movies_data(query, limit=8)
    return jsonify({"suggestions": suggestions})


if __name__ == "__main__":
    app.run(debug=True)