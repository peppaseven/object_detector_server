from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
if sys.version_info.major == 2:
    import cStringIO as StringIO
    import urllib2
else:
    from io import StringIO
    from io import BytesIO
    from urllib import request
    import base64

import numpy as np
import os
import re
import time
import datetime
import logging
import flask
import werkzeug
import optparse

import math
from utils import exifutil
import tensorflow as tf
from PIL import Image
from fileinput import filename

from utils import label_map_util
from utils import visualization_utils as vis_util

REPO_DIRNAME = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + '/data')
MODEL_DIRNAME =  os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + '/models')
UPLOAD_FOLDER = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + '/img_uploads')
DETECTED_FOLDER = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + '/img_detected')

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

# Obtain the flask app object
app = flask.Flask(__name__)

NUM_CLASSES = 90
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpe', 'jpeg'])

FLAGS = tf.app.flags.FLAGS

@app.route('/detect_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    localfile = '/tmp/samplefile.png'
    try:
        bytes = request.urlopen(imageurl).read()
        #save to tmp,just for temp use for my classify api.

        tmpfile = open(localfile, 'wb')
        tmpfile.write(bytes)
        tmpfile.close();

        if sys.version_info.major == 2:
            string_buffer = StringIO.StringIO(bytes)
        else:
            string_buffer = BytesIO(bytes)
        image = exifutil.open_oriented_im(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'detection.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    app.logger.info('Image: %s', imageurl)

    image_np, time_cost = app.clf.detect_image(localfile)
    os.remove(localfile)
    return flask.render_template(
        'detection.html', has_result=True, result=[True, '%.3f' % time_cost],
        imagesrc=embed_image_html(image)
    )

@app.route('/detect_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        path, extension = os.path.splitext(filename)
        if extension == '.png':
            im = Image.open(filename)
            filename = "%s.jpg" % path
            im.save(filename)

        logging.info('Saving to %s.', filename)



    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    image_np,time_cost = app.clf.detect_image(filename)
    outfile = os.path.join(DETECTED_FOLDER,filename_)
    app.clf.save_png_image(image_np,outfile)
    image = exifutil.open_oriented_im(outfile)

    return flask.render_template(
        'detection.html', has_result=True, result=[True,'%.3f' % time_cost],
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))

    image_pil = image_pil.resize((256, 256))
    if sys.version_info.major == 2:
        string_buf=StringIO.StringIO()
        image_pil.save(string_buf, format='png')
        data = string_buf.getvalue().encode('base64').replace('\n', '')
    else:
        _buf = BytesIO()
        image_pil.save(_buf, format='png')
        _buf.seek(0)
        b64_buf = base64.b64encode(_buf.getvalue())
        string_buf = StringIO(b64_buf.decode('utf-8', errors='replace'))
        data =string_buf.getvalue().replace('\n', '')

    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS =  'mscoco_label_map.pbtxt'


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


class ObjectDetection(object):
    default_args = {
        'model_graph_file': (
            '{}/{}'.format(MODEL_DIRNAME,PATH_TO_CKPT)),
        'label_map_file': (
            '{}/{}'.format(REPO_DIRNAME,PATH_TO_LABELS)),
    }
    for key, val in default_args.items():

        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))

    def __init__(self, model_graph_file, label_map_file):
        logging.info('Loading net and associated files...')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.detection_graph)
        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(label_map_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


    def detect_image(self, image):
        try:
            start_time = time.time()
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            if not tf.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            print('Path is %s' %(image))
            image_data = Image.open(image)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image_data)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection.
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            end_time = time.time()
            app.logger.info("detection cost %.2f secs", end_time - start_time)
            return image_np, end_time - start_time
        except Exception as err:
            logging.info('Object detection error: %s', err)
            return None

    def save_png_image(self,image_np,output_path):
         vis_util.save_image_array_as_png(image_np,output_path)


def setup_app(app):
    app.clf = ObjectDetection(**ObjectDetection.default_args)
    app.logger.info('this is for warmup...')
    sample = os.path.join(REPO_DIRNAME, "sample.jpg")
    ret,time_cost = app.clf.detect_image(sample)
    app.logger.info("sample testing complete %.3f",time_cost)

def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=31000)
    opts, args = parser.parse_args()
    # Initialize classifier + warm start by forward for allocation
    setup_app(app)
    app.run(debug=True, processes=1, host='0.0.0.0', port=opts.port)

logging.getLogger().setLevel(logging.INFO)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DETECTED_FOLDER):
    os.makedirs(DETECTED_FOLDER)

if __name__ == '__main__':
    start_from_terminal(app)
else:
    gunicorn_error_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_error_logger.handlers
    app.logger.setLevel(logging.INFO)
    setup_app(app)
