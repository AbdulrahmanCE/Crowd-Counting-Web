from django.db import models


# Create your models here.


class Image(models.Model):
    image = models.ImageField(upload_to='image')


class Density(models.Model):
    density = models.ImageField(upload_to='image')


class Ground(models.Model):
    ground = models.ImageField(upload_to='image')


class Test(models.Model):
    test = models.ImageField(upload_to='test')
