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
import json
import math
from utils import exifutil
import tensorflow as tf
from PIL import Image
from fileinput import filename
from utils.tempimage import TempImage
REPO_DIRNAME = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + '/data')
UPLOAD_FOLDER = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + '/img_uploads')
	
NUM_CLASSES = 5
NUM_TOP_CLASSES = 5
# Obtain the flask app object
app = flask.Flask(__name__)


ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpe', 'jpeg'])

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

@app.route('/', methods=['GET', 'POST'])
def classify_index():
    string_buffer = None

    if flask.request.method == 'GET':
        url = flask.request.args.get('url')
        if url:
            logging.info('Image: %s', url)
            string_buffer = request.urlopen(url).read()

        file = flask.request.args.get('file')
        if file:
            logging.info('Image: %s', file)
            string_buffer = open(file, 'rb').read()

        if not string_buffer:
            return flask.render_template('classify.html', has_result=False)

    elif flask.request.method == 'POST':
        string_buffer = flask.request.stream.read()

    if not string_buffer:
        resp = flask.make_response()
        resp.status_code = 400
        return resp
    names, time_cost, accuracy = app.clf.classify_image(file)
    return flask.make_response(u",".join(names), 200, {'ClassificationAccuracy': accuracy})


@app.route('/classify_url', methods=['GET'])
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
            'classify.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    app.logger.info('Image: %s', imageurl)

    names, time_cost, probs = app.clf.classify_image(localfile)
    os.remove(localfile)
    return flask.render_template(
        'classify.html', has_result=True, result=[True, zip(names, probs), '%.3f' % time_cost],
        imagesrc=embed_image_html(image)
    )


@app.route('/classify_upload', methods=['POST'])
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
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'classify.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    names,time_cost, probs = app.clf.classify_image(filename)

    return flask.render_template(
        'classify.html', has_result=True, result=[True, zip(names, probs), '%.3f' % time_cost],
        imagesrc=embed_image_html(image)
    )

@app.route('/classify_image', methods=['POST'])
def classify_image():
    t = TempImage()
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        # write the image to temporary file

        imagefile.save(t.path)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return json.dumps({'has_result':False})

    names,time_cost, probs = app.clf.classify_image(t.path)
    t.cleanup()
    return json.dumps({'has_result':True,'name':names[0]})

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


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path,
                 uid_lookup_path):

        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

class ImagenetClassifier(object):
    default_args = {
        'model_graph_file': (
            '{}/classify_image_graph_def.pb'.format(REPO_DIRNAME)),
        'label_map_file': (
            '{}/imagenet_2012_challenge_label_map_proto.pbtxt'.format(REPO_DIRNAME)),
        'human_label_map': (
            '{}/imagenet_synset_to_human_label_map.txt'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.items():

        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))

    def __init__(self, model_graph_file, label_map_file,human_label_map):
        logging.info('Loading net and associated files...')
        with tf.gfile.FastGFile(model_graph_file,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            self.sess = tf.Session()
            self.node_lookup = NodeLookup(label_map_file,human_label_map)

    def eval_image(self, image, height, width, scope=None):
        """Prepare one image for evaluation.

        Args:
          image: 3-D float Tensor
          height: integer
          width: integer
          scope: Optional scope for op_scope.
        Returns:
          3-D float Tensor of prepared image.
        """
        #image = tf.reshape(image, [height,width,3])
        # return images
        with tf.op_scope([image, height, width], scope, 'eval_image'):
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            image = tf.image.central_crop(image, central_fraction=0.875)

            # Resize the image to the original height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
            return image

    def classify_image(self, image):
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
            image_data = tf.gfile.FastGFile(image, 'rb').read()
            softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')
            predictions = self.sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})

            predictions = np.squeeze(predictions)
            # Creates node ID --> English string lookup.
            top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
            label_names = []
            probs = []
            for node_id in top_k:
                human_string = self.node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                label_names.append(human_string.split(',')[0])
                probs.append(score)
                app.logger.info('%s (score = %.5f)' % (human_string, score))


            end_time = time.time()
            app.logger.info("classify_image cost %.2f secs", end_time - start_time)
            return label_names[:3], end_time - start_time,probs[:3]
        except Exception as err:
            logging.info('Classification error: %s', err)
            return None

def setup_app(app):
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.logger.info('this is for warmup...')
    sample = os.path.join(REPO_DIRNAME, "sample.jpg")
    ret,_,_ = app.clf.classify_image(sample)
    app.logger.info("sample testing complete %s %s %s", ret[0], ret[1], ret[2])

def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5005)
    opts, args = parser.parse_args()
    # Initialize classifier + warm start by forward for allocation
    setup_app(app)
    app.run(debug=True, processes=1, host='0.0.0.0', port=opts.port)

logging.getLogger().setLevel(logging.INFO)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
    start_from_terminal(app)
else:
    gunicorn_error_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_error_logger.handlers
    app.logger.setLevel(logging.INFO)
    setup_app(app)
