import os
import tarfile
import numpy as np
from PIL import Image
import cv2

import warnings

warnings.filterwarnings("ignore")

# %tensorflow_version 1.x
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        self.n_cpus = 20

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(
            config=tf.ConfigProto(device_count={"CPU": self.n_cpus}, inter_op_parallelism_threads=self.n_cpus,
                                  intra_op_parallelism_threads=10), graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


MODEL = DeepLabModel("deeplab_model.tar.gz")
print('model loaded successfully!')


def mask_model(frame):
    # Changing opencv BGR format to pillow supported RGB format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # converting image array to pillow image object
    image = Image.fromarray(img)
    # giving pillow image to model
    resized_im, seg_map = MODEL.run(image)
    # converting "resized_im" pillow object to numpy array
    resized_im = np.array(resized_im)
    # Changing pillow supported RGB format to opencv BGR format
    resized_im = cv2.cvtColor(resized_im, cv2.COLOR_RGB2BGR)
    # converting model generated segmentation map into a color_map
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    # resizing the image output and segmentation output in to same size
    seg_image = cv2.resize(seg_image, resized_im.shape[1::-1])
    # Detecting the persons color and removing remaining colors in segmentation
    lower = np.array([192, 128, 128], dtype="uint8")
    upper = np.array([192, 128, 128], dtype="uint8")
    mask = cv2.inRange(seg_image, lower, upper)
    seg_output = cv2.bitwise_and(resized_im, resized_im, mask=mask)
    return (resized_im,mask,seg_output)


frame = cv2.imread('girl.jpg')
out = mask_model(frame)
gray=cv2.cvtColor(out[0],cv2.COLOR_RGB2GRAY)
gray= cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
gray = cv2.bitwise_and(gray, gray, mask=255-out[1])
orgframe=cv2.resize(frame,(out[0].shape[1],out[0].shape[0]))
splashframe=cv2.add(gray,out[2])
maskframe=out[2]
cv2.imshow('COLOR SPLASH EFFECT',np.hstack([orgframe,maskframe,splashframe]))
cv2.waitKey(20000)