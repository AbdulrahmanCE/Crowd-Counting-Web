import os
import glob
import shutil
import cv2

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

from crowd_web import settings
from .models import Image, Density, Ground, Test
from . import predict_file


# Create your views here.


def crowd(request):
    return render(request, 'crowdcounting.html')


def predict_func(request):
    img = None
    dm = None

    if request.method == "POST":
        # shutil.rmtree('media')
        # os.makedirs('media')

        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        img = Image(image=request.FILES['image'])
        img_path = os.path.join('media', uploaded_file.name)
        print(img_path)
        predict_file.predict([img_path])
        dm = Density(density='DM.png')

        # os.remove('media/{}'.format(uploaded_file.name))
    return render(request, 'predictCrowd.html', {'img': img, 'dm': dm})


def density(request):
    img = None
    dm = None
    pred = None
    test = None
    flag = True

    if request.method == "POST":
        # shutil.rmtree('media')
        # os.makedirs('media')

        img_file = request.FILES['image']
        dm_file = request.FILES['ground-truth']
        im = FileSystemStorage()
        dem = FileSystemStorage()
        im.save(img_file.name, img_file)
        dem.save(dm_file.name, dm_file)

        img = Image(image=img_file)

        img_path = os.path.join('media', img_file.name)
        dm_path = os.path.join('media', dm_file.name)

        predict_file.calculatedesnity([img_path], [dm_path])

        predict_file.predict([img_path])
        dm = Ground(ground='GT.png')
        pred = Density(density='DM.png')
    test = []
    for x in range(1, 9):
        print(x)
        test.append(Test(test='test/image{}.jpg'.format(x)))

        # os.remove('media/{}'.format(img_file.name))
    # httprespons
    return render(request, 'calculateDM.html', {'img': img, 'dm': dm, 'pred': pred, 'testimages': test, 'flag': flag})


def testGT(request, num):
    img_path = 'static/img/image{}.jpg'.format(num)
    dm_path = 'static/img/image{}.mat'.format(num)

    predict_file.calculatedesnity([img_path], [dm_path])
    predict_file.predict([img_path])

    img = Image(image='test/image{}.jpg'.format(num))
    dm = Ground(ground='GT.png')
    pred = Density(density='DM.png')

    # os.remove('media/{}'.format(img_file.name))
    return render(request, 'calculateDM.html', {'img': img, 'dm': dm, 'pred': pred})


def about(request):
    return render(request, 'aboutus.html')


def contact(request):
    return render(request, 'contactus.html')
