from django.db import models
from django.contrib.auth.models import AbstractUser


# Create your models here.
class Embeds(models.Model):
    name = models.CharField(max_length=100)
    id = models.PositiveIntegerField(primary_key=True)
    embed1 = models.CharField(max_length=1000)
    embed2 = models.CharField(max_length=1000)
    embed3 = models.CharField(max_length=1000)

    def __str__(self):
        return f"Embeds for {self.name}"
    

class Profile(models.Model):
    name = models.CharField(max_length=100)
    rank = models.CharField(max_length=100)
    service_no = models.PositiveIntegerField(primary_key=True)

    def __str__(self):
        return f"{self.name}"
    

# class CustomUser(AbstractUser):
#     service_no = models.CharField(max_length=100)