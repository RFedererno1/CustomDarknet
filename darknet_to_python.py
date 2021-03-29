import darknet
from ctypes import *
import cv2

# CONFIG_PATH = 'darknet_cfg/yolo-obj.cfg'
# DATA_PATH = 'darknet_cfg/obj1.data'
# WEIGHT_PATH = '/content/yolo-obj_best_no_cudnn_half.weights'

# network, class_names, _ = darknet.load_network(CONFIG_PATH, DATA_PATH, WEIGHT_PATH, batch_size=1)
class Detect:
  def __init__(self, CONFIG_PATH, DATA_PATH, WEIGHT_PATH):
    self.CONFIG_PATH = CONFIG_PATH
    self.DATA_PATH = DATA_PATH
    self.WEIGHT_PATH = WEIGHT_PATH
    self.CONF_THRESH = 0.25
    self.network, self.class_names, _ = darknet.load_network(self.CONFIG_PATH, self.DATA_PATH, self.WEIGHT_PATH, batch_size=1)

  def detect_obj(self, image):
    width = darknet.network_width(self.network)
    height = darknet.network_height(self.network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    original_image = darknet.make_image(w, h, 3)
    darknet.copy_image_from_bytes(original_image, image_rgb.tobytes())
    resized_image = darknet.letterbox_image(original_image, width, height)
    detections = darknet.detect_image(self.network, self.class_names, resized_image, thresh=self.CONF_THRESH, w = w, h = h)
    darknet.free_image(original_image)
    darknet.free_image(resized_image)
    detections = [(x[0],x[1],x[2],cv2.cvtColor(image[int(x[2][1] - x[2][3]/2):int(x[2][1] + x[2][3]/2),int(x[2][0] - x[2][2]/2):int(x[2][0] + x[2][2]/2)], cv2.COLOR_BGR2RGB)) for x in detections]
    return detections
