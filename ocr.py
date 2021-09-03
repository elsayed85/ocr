
import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time
import json
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import numpy


def kmeans(input_img, k, i_val):
    hist = cv2.calcHist([input_img], [0], None, [256], [0, 256])
    img = input_img.ravel()
    img = np.reshape(img, (-1, 1))
    img = img.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        img, k, None, criteria, 10, flags)
    centers = np.sort(centers, axis=0)

    return centers[i_val].astype(int), centers, hist


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
original_image = cv2.imread('car2.jpg')
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
thresh = 255 - cv2.threshold(image, int(kmeans(input_img=image,
                             k=8, i_val=2)[0]), 255, cv2.THRESH_BINARY)[1]

x, y, w, h = 300, 275, 120, 40


# deifne areas location and label for each area
areas = [
    {
        "x": 300,
        "y": 275,
        "w": 120,
        "h": 40,
        "name": "A01"
    },
]

#x,y,w,h = 349,33,40,25
#x, y, w, h = 107, 592, 40, 25

NUM_CLUSTERS = 5

output = {}
for area in areas:
    ROI = thresh[y:area["y"]+area["h"], x:area["x"]+area["w"]]
    ROI_ORIGNAL = original_image[y:area["y"]+area["h"], x:area["x"]+area["w"]]
    chars = pytesseract.image_to_string(ROI, lang='eng',  config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

    if(chars):
        cv2.imwrite("output.jpg", ROI_ORIGNAL)
        im = Image.open('output.jpg')
        im = im.resize((150, 150))      # optional, to reduce time
        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

        print('finding clusters')
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        print('cluster centres:\n', codes)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = numpy.histogram(vecs, len(codes))    # count occurrences

        # find most frequent
        index_max = scipy.argmax(counts)
        peak = codes[index_max]
        colour = binascii.hexlify(bytearray(int(c)
                                  for c in peak)).decode('ascii')
        output[area["name"]] = {
            "colour": colour,
            "chars": chars.strip()
        }

print(json.dumps(output))
