from styx_msgs.msg import TrafficLight
import tensorflow as tf
import os
import sys
import cv2
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.PROB_THRESHOLD = 0.5
        self.IOU_THRESHOLD = 0.05
        self.classId = ('Green', 'Red', 'Yellow') 
        self.CURR_DIR = os.getcwd()
        self.MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/fasterRCNNV2_sim_site_pb_model_10000_rank_1/frozen_inference_graph.pb'
        
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(self.MODEL_PATH, 'rb') as fid:        
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')        
        
        for n in tf.get_default_graph().as_graph_def().node:
            print(n.name)
            
        self.sess = tf.Session(graph=detection_graph)
        
#         with detection_graph.as_default():
#             with tf.Session(graph=detection_graph) as sess:
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        print('TL Classifier loaded ...')
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # detect lights
        image_expanded = np.expand_dims(image, axis=0)
        time0 = time.time()
        boxes, scores, classes, num = self.sess.run( [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        
        #print ('\nprocessing time: {:.2f}s'.format(time.time() - time0))
        
        # threshold detectected bixes
        boxes_thresholded = []
        scores_thresholded = []
        classes_thresholded = []
        for i in range(boxes.shape[1]):
            if scores[0][i] > self.PROB_THRESHOLD:
                #print('{}: {:.2f}%'.format(self.classId[int(classes[0][i])-1], scores[0][i]*100))
                boxes_thresholded.append(boxes[0][i])
                scores_thresholded.append(scores[0][i])
                classes_thresholded.append(classes[0][i])
        
        #print('classes_thresholded \n{}'.format(classes_thresholded))
        #print('scores_thresholded \n{}'.format(scores_thresholded))
        
        # if there are detections
        if len(scores_thresholded) > 0:
            # check if there is overlapping box,
            # if two boxes overlap keep the one with higher confidence            
            for i in range(len(boxes_thresholded)):
                for j in range(len(boxes_thresholded)):
                    if i != j:
                        iou = self.check_overlap_fnc(boxes_thresholded[i], boxes_thresholded[j])
                        if iou > self.IOU_THRESHOLD:
                            if scores_thresholded[i] >= scores_thresholded[j]:
                                boxes_thresholded.pop(j)
                                scores_thresholded.pop(j)
                                classes_thresholded.pop(j)
                            elif scores_thresholded[i] < scores_thresholded[j]:
                                boxes_thresholded.pop(i)
                                scores_thresholded.pop(i)
                                classes_thresholded.pop(i)                            
            
            # count the majority
            lightCount = [0, 0, 0] # green, red, yellow
            for i in range(len(classes_thresholded)):
                if classes_thresholded[i] == 1:
                    lightCount[0] += 1
                if classes_thresholded[i] == 2:
                    lightCount[1] += 1
                if classes_thresholded[i] == 3:
                    lightCount[2] += 1
            
            #print('lightCount {}'.format(lightCount))
            
            #self.draw_a_detection_result( boxes, scores, classes, num, image)
            
            max_light = lightCount.index(max(lightCount) )
            if max_light == 0:
                #print('light is : green')
                return TrafficLight.GREEN
            if max_light == 1:
                #print('light is : red')
                return TrafficLight.RED
            if max_light == 2:
                #print('light is : yellow')
                return TrafficLight.YELLOW
            else:
                return TrafficLight.UNKNOWN
        else:
                return TrafficLight.UNKNOWN
    
    def check_overlap_fnc(self, bb1, bb2):
        assert bb1[0] < bb1[2]
        assert bb1[1] < bb1[3]
        assert bb2[0] < bb2[2]
        assert bb2[1] < bb2[3]
    
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
    
        if x_right < x_left or y_bottom < y_top:
            return 0.0
    
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def draw_a_detection_result(self, boxes, scores, classes, num, image_np):
        sys.path.append('/home/student/models/research/object_detection/')
        from utils import label_map_util
        from utils import visualization_utils as vis_util
        
        label_map = label_map_util.load_labelmap(os.path.dirname(os.path.realpath(__file__)) + '/label_map.pbtxt')
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=3, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        
        is_plot_result = False
        is_make_video = True
        min_score_thresh = .30

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np, boxes, classes, scores,
            category_index,
            min_score_thresh=min_score_thresh,
            use_normalized_coordinates=True,
            line_thickness=3)
    

        plt.figure(figsize=(12, 8))
        plt.imshow(image_np)
        plt.show()
    

if __name__ == '__main__':
    classifier = TLClassifier()
    #check_iou = classifier.check_overlap_fnc([0, 0, 0.75, 0.75],[0.5, 0.5, 1.0, 1.0])
    #print('IOU should be 0.08333, got: {:.5f} check passed'.format(check_iou))
    
    check_images = ['img_0.jpg', 'img_759.jpg', 'img_2654.jpg']
    #check_images = ['img_0.jpg', 'left0700.jpg', 'left0704.jpg']
    for i in range(len(check_images)):
        image_path = os.getcwd() + r'/train_data/' + check_images[i]
        print('\n' + image_path)
        img = Image.open(image_path)
        (im_width, im_height) = img.size
        img_np = np.array(img.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        light = classifier.get_classification(img_np)
