import darknet
from ctypes import *
import cv2

# CONFIG_PATH = 'darknet_cfg/yolo-obj.cfg'
# DATA_PATH = 'darknet_cfg/obj1.data'
# WEIGHT_PATH = '/content/yolo-obj_best_no_cudnn_half.weights'
CONF_THRESH = 0.25

# network, class_names, _ = darknet.load_network(CONFIG_PATH, DATA_PATH, WEIGHT_PATH, batch_size=1)

def detect_obj(image, network, class_names):
  width = darknet.network_width(network)
  height = darknet.network_height(network)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  h, w, _ = image_rgb.shape
  original_image = darknet.make_image(w, h, 3)
  darknet.copy_image_from_bytes(original_image, image_rgb.tobytes())
  resized_image = darknet.letterbox_image(original_image, width, height)
  detections = darknet.detect_image(network, class_names, resized_image, thresh=CONF_THRESH, w = w, h = h)
  darknet.free_image(original_image)
  darknet.free_image(resized_image)
  detections = [(x[0],x[1],x[2],cv2.cvtColor(image[int(x[2][1] - x[2][3]/2):int(x[2][1] + x[2][3]/2),int(x[2][0] - x[2][2]/2):int(x[2][0] + x[2][2]/2)], cv2.COLOR_BGR2RGB)) for x in detections]
  return detections
