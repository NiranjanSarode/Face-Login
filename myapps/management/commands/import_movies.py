import csv
from django.core.management.base import BaseCommand
from myapps.models import Movie

class Command(BaseCommand):
    help = 'Imports movie data from a CSV file into the database'

    def add_arguments(self, parser):
        parser.add_argument('imdb_top_1000.csv', type=str, help='imdb_top_1000.csv')

    def handle(self, *args, **kwargs):
        csv_file_path = kwargs['imdb_top_1000.csv']
        
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            movie_data = []
            for row in reader:
                # Check if meta_score is empty
                meta_score = row['Meta_score']
                if meta_score:
                    meta_score = float(meta_score)
                else:
                    meta_score = None  # or any default value you prefer

                movie_instance = Movie(
                    poster_link=row['Poster_Link'],
                    series_title=row['Series_Title'],
                    released_year=row['Released_Year'],
                    certificate=row['Certificate'],
                    runtime=row['Runtime'],
                    genre=row['Genre'],
                    imdb_rating=float(row['IMDB_Rating']),
                    overview=row['Overview'],
                    meta_score=meta_score,
                    director=row['Director'],
                    star1=row['Star1'],
                    star2=row['Star2'],
                    star3=row['Star3'],
                    star4=row['Star4'],
                    no_of_votes=int(row['No_of_Votes']),
                    gross=row['Gross']
                )
                movie_data.append(movie_instance)

            Movie.objects.bulk_create(movie_data)
        
        self.stdout.write(self.style.SUCCESS('Successfully imported data from CSV'))