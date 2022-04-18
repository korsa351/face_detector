# Create your views here.
# import the necessary packages
from django.shortcuts import render
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import cv2, os, string, io, random, urllib.request

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
global out_image


def generate_random_string(length=5):
    letters = string.ascii_lowercase
    rand_string = ''.join(random.choice(letters) for i in range(length))
    return rand_string


def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    print(type(io_buf))
    return io_buf.read()


@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}
    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("Image", None) is not None:
            # grab the uploaded image
            x = request.FILES["Image"]
            image = _grab_image(stream=request.FILES["Image"])
            out_image = x.file
            out_image = Image.open(x.file)
        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.GET.get("url", None)
            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
            # load the image and convert
            image = _grab_image(url=url)
        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
        # update the data dictionary with the faces detected
        data.update({"num_faces": len(rects), "faces": rects, "success": True})
        out_image = np.array(out_image)
        for i in rects:
            out_image = cv2.rectangle(out_image, i[0:2], i[-2::], (255, 0, 0), thickness=2)
        out_image = Image.fromarray(out_image)
        string = generate_random_string()
        out_image.save(f'media/photo_{string}.png')
        context = {'url': f'/media/photo_{string}.png'}
        return render(request, 'result.html', context=context)
    return render(request, 'upload.html')


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image
