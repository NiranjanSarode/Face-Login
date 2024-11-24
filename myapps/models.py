# Create your models here.
from django.db import models
# from myapps.models import WindData

from django.db import models

class Artist(models.Model):
    artistid = models.AutoField(primary_key=True)
    name = models.CharField(max_length=120, null=True, blank=True)

    class Meta:
        db_table = 'Artist'
        managed = False

class Album(models.Model):
    albumid = models.AutoField(primary_key=True)
    title = models.CharField(max_length=160)
    artistid = models.ForeignKey(Artist, on_delete=models.CASCADE, db_column='ArtistId')

    class Meta:
        db_table = 'Album'
        managed = False

class Customer(models.Model):
    customerid = models.AutoField(primary_key=True)
    firstname = models.CharField(max_length=40)
    lastname = models.CharField(max_length=20)
    company = models.CharField(max_length=80, null=True, blank=True)
    address = models.CharField(max_length=70, null=True, blank=True)
    city = models.CharField(max_length=40, null=True, blank=True)
    state = models.CharField(max_length=40, null=True, blank=True)
    country = models.CharField(max_length=40, null=True, blank=True)
    postalcode = models.CharField(max_length=10, null=True, blank=True)
    phone = models.CharField(max_length=24, null=True, blank=True)
    fax = models.CharField(max_length=24, null=True, blank=True)
    email = models.CharField(max_length=60)
    supportrepid = models.ForeignKey('Employee', null=True, blank=True, on_delete=models.SET_NULL, db_column='SupportRepId')

    class Meta:
        db_table = 'Customer'
        managed = False

class Employee(models.Model):
    employeeid = models.AutoField(primary_key=True)
    lastname = models.CharField(max_length=20)
    firstname = models.CharField(max_length=20)
    title = models.CharField(max_length=30, null=True, blank=True)
    reportsto = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, db_column='ReportsTo')
    birthdate = models.DateTimeField(null=True, blank=True)
    hiredate = models.DateTimeField(null=True, blank=True)
    address = models.CharField(max_length=70, null=True, blank=True)
    city = models.CharField(max_length=40, null=True, blank=True)
    state = models.CharField(max_length=40, null=True, blank=True)
    country = models.CharField(max_length=40, null=True, blank=True)
    postalcode = models.CharField(max_length=10, null=True, blank=True)
    phone = models.CharField(max_length=24, null=True, blank=True)
    fax = models.CharField(max_length=24, null=True, blank=True)
    email = models.CharField(max_length=60, null=True, blank=True)

    class Meta:
        db_table = 'Employee'
        managed = False

class Genre(models.Model):
    genreid = models.AutoField(primary_key=True)
    name = models.CharField(max_length=120, null=True, blank=True)

    class Meta:
        db_table = 'Genre'
        managed = False

class Invoice(models.Model):
    invoiceid = models.AutoField(primary_key=True)
    customerid = models.ForeignKey(Customer, on_delete=models.CASCADE, db_column='CustomerId')
    invoicedate = models.DateTimeField()
    billingaddress = models.CharField(max_length=70, null=True, blank=True)
    billingcity = models.CharField(max_length=40, null=True, blank=True)
    billingstate = models.CharField(max_length=40, null=True, blank=True)
    billingcountry = models.CharField(max_length=40, null=True, blank=True)
    billingpostalcode = models.CharField(max_length=10, null=True, blank=True)
    total = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        db_table = 'Invoice'
        managed = False

class InvoiceLine(models.Model):
    invoicelineid = models.AutoField(primary_key=True)
    invoiceid = models.ForeignKey(Invoice, on_delete=models.CASCADE, db_column='InvoiceId')
    trackid = models.ForeignKey('Track', on_delete=models.CASCADE, db_column='TrackId')
    unitprice = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.IntegerField()

    class Meta:
        db_table = 'InvoiceLine'
        managed = False

class MediaType(models.Model):
    mediatypeid = models.AutoField(primary_key=True)
    name = models.CharField(max_length=120, null=True, blank=True)

    class Meta:
        db_table = 'MediaType'
        managed = False

class Playlist(models.Model):
    playlistid = models.AutoField(primary_key=True)
    name = models.CharField(max_length=120, null=True, blank=True)

    class Meta:
        db_table = 'Playlist'
        managed = False

class PlaylistTrack(models.Model):
    playlistid = models.ForeignKey(Playlist, on_delete=models.CASCADE, db_column='PlaylistId')
    trackid = models.ForeignKey('Track', on_delete=models.CASCADE, db_column='TrackId')

    class Meta:
        db_table = 'PlaylistTrack'
        managed = False
        unique_together = (('playlistid', 'trackid'),)

class Track(models.Model):
    trackid = models.AutoField(primary_key=True)
    name = models.CharField(max_length=200)
    albumid = models.ForeignKey(Album, null=True, blank=True, on_delete=models.SET_NULL, db_column='AlbumId')
    mediatypeid = models.ForeignKey(MediaType, on_delete=models.CASCADE, db_column='MediaTypeId')
    genreid = models.ForeignKey(Genre, null=True, blank=True, on_delete=models.SET_NULL, db_column='GenreId')
    composer = models.CharField(max_length=220, null=True, blank=True)
    milliseconds = models.IntegerField()
    bytes = models.IntegerField(null=True, blank=True)
    unitprice = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        db_table = 'Track'
        managed = False




# class WindData(models.Model):
#     id = models.AutoField(primary_key=True)  # Explicitly set the id as the primary key
#     wind_speed = models.FloatField()
#     wind_bearing = models.IntegerField()
#     visibility = models.FloatField()
#     loud_cover = models.IntegerField()

#     def __str__(self):
#         return f"{self.wind_speed} km/h at {self.wind_bearing} degrees"



# class WeatherData(models.Model):
#     formatted_date = models.CharField(max_length=255)
#     summary = models.CharField(max_length=255)
#     precip_type = models.CharField(max_length=100, default='unknown')
#     temperature = models.FloatField()
#     apparent_temperature = models.FloatField()
#     humidity = models.FloatField()
#     pressure = models.FloatField()
#     daily_summary = models.TextField()
#     wind= models.ForeignKey(WindData, on_delete=models.CASCADE)

#     def __str__(self):
#         return self.formatted_date





# class Inventory(models.Model):
#     product_name=models.CharField(max_length=30,null=False,blank=False)
#     cost_per_item=models.DecimalField(max_digits=12,decimal_places=2,null=False,blank=False)
#     quantity_in_stock=models.IntegerField(null=False,blank=False)
#     quantity_sold=models.IntegerField(null=False,blank=False)
#     sales=models.DecimalField(max_digits=12,decimal_places=2,null=False,blank=False)
#     stock_date=models.DateField()
#     photos=models.ImageField(upload_to="Inventph/")

#     def __str__(self):
#         return self.product_name

# class Weather(models.Model):
#     formatted_date = models.DateTimeField()
#     summary = models.CharField(max_length=200)
#     precip_type = models.CharField(max_length=50, null=True, blank=True)
#     temperature_c = models.FloatField()
#     apparent_temperature_c = models.FloatField()
#     humidity = models.FloatField()
#     wind_speed_kmh = models.FloatField()
#     wind_bearing_degrees = models.IntegerField()
#     visibility_km = models.FloatField()
#     loud_cover = models.FloatField()
#     pressure_millibars = models.FloatField()
#     daily_summary = models.TextField()

#     def __str__(self):
#         return self.precip_type

# class Movie(models.Model):
#     poster_link = models.URLField()
#     series_title = models.CharField(max_length=255)
#     released_year = models.CharField(max_length=10)
#     certificate = models.CharField(max_length=10)
#     runtime = models.CharField(max_length=50)
#     genre = models.CharField(max_length=255)
#     imdb_rating = models.FloatField()
#     overview = models.TextField()
#     meta_score = models.CharField(max_length=10, null=True)
#     director = models.CharField(max_length=255)
#     star1 = models.CharField(max_length=255)
#     star2 = models.CharField(max_length=255)
#     star3 = models.CharField(max_length=255)
#     star4 = models.CharField(max_length=255)
#     no_of_votes = models.IntegerField()
#     gross = models.CharField(max_length=255)  # Gross is a string due to commas and other characters

#     def __str__(self):
#         return self.series_title

    
class QueryHistory(models.Model):
    user_id = models.IntegerField()
    query_history = models.JSONField()


    def __str__(self):
        return self.user_id
    
    