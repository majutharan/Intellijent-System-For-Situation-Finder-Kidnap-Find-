from django.db import models


# Create your models here.

class Video(models.Model):
    uid = models.CharField(max_length=50)
    video = models.FileField(upload_to='upload/videos')

    class Meta:
        db_table = "videos"


class Audio(models.Model):
    uid = models.CharField(max_length=50)
    audio = models.FileField(upload_to='upload/audios')

    class Meta:
        db_table = "Audio"


class User(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    name = models.CharField(max_length=100)
    age = models.CharField(max_length=100)
    place = models.CharField(max_length=100)
    work = models.CharField(max_length=100)
    sex = models.CharField(max_length=100)
    mail_id = models.CharField(max_length=100)

    class Meta:
        db_table = "User"
