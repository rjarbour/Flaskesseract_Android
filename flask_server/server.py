from flask import render_template
from flask import request
from flask import Response
from flask import Flask
app = Flask(__name__)

# import the necessary packages
import pytesseract
import cv2
import numpy as np
import io
import werkzeug
import tempfile


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def compute_skew(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape
    # Filter removed
    edges = auto_canny(image)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=width / 2.0, maxLineGap=50)
    angle = 0.0
    # nlines = lines.size
    nlines = lines.shape[0]
    lines = lines.reshape(lines.shape[0], 4)
    # for x1, y1, x2, y2 in lines[0]:
    for x1, y1, x2, y2 in lines:
        angle += np.arctan2(y2 - y1, x2 - x1)

    # return angle / nlines
    angle /= nlines
    return angle * 180 / np.pi


def deskew(image, angle):
    image = cv2.bitwise_not(image)
    non_zero_pixels = cv2.findNonZero(image)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = image.shape
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    return cv2.getRectSubPix(rotated, (cols, rows), center)


def process(f):
    # load the example image and convert it to grayscale
    (b, g, r) = cv2.split(f)
    deskewed_image = deskew(r.copy(), compute_skew(r))
    deskewed_image_canny = auto_canny(deskewed_image)
    dilated = cv2.dilate(deskewed_image, np.ones((20, 40), np.uint8))
    ret, thresh = cv2.threshold(dilated, 244, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # dilated = cv2.drawContours(dilated, contours,-1,(0,255,0),3)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cv2.rectangle(deskewed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    deskewed_image = deskewed_image[y:y + h, x:x + w]

    #filename = "{}.png".format(os.getpid())
    #cv2.imwrite(filename, deskewed_image)
    return pytesseract.image_to_string(deskewed_image)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == "GET":
        return render_template("index.html")


@app.route('/uploader', methods=['POST'])
def hello():
    def custom_stream_factory(total_content_length, filename, content_type, content_length=None):
        tmpfile = tempfile.NamedTemporaryFile('wb+', prefix='flaskapp')
        return tmpfile

    stream, form, files = werkzeug.formparser.parse_form_data(request.environ,
                                                              stream_factory=custom_stream_factory)

    for fil in files.values():
        print(
            " ".join(["saved form name", fil.name, "submitted as", fil.filename, "to temporary file", fil.stream.name]))
        nparr = np.fromstring(fil.stream.read(), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return process(img_np)

if __name__ == '__main__':
    app.run(debug=True)
