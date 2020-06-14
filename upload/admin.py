from django.contrib import admin
from upload.models import Image, Density

# Register your models to admin site, then you can add, edit, delete and search your models in Django admin site.
admin.site.register(Image)
admin.site.register(Density)

