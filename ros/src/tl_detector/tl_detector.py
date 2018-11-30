#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
<<<<<<< HEAD
from scipy.spatial import KDTree
=======
import math
import time
import random
import numpy as np
import cPickle as pickle
>>>>>>> a0dfde1cadd07fca35b9e603c4b65d834f8ef87e

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.MODE = 'SIMULATION'
        self.pose = None
        self.waypoints = None
        self.camera_image = None
<<<<<<< HEAD
        self.waypoints_2d = None
        self.waypoints_tree = None
        
        self.lights = []
        self.traffic_light_wp_indexes = []
=======
        self.waypoints_2d  = None
        self.waypoint_tree = None
        self.ego_wp_idx = None
        self.lights = []
        self.traffic_ego_dist = None
        self.light_state = -1
>>>>>>> a0dfde1cadd07fca35b9e603c4b65d834f8ef87e

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.RUNNING_MODE = 'SIMULATION'     # Mode of running, SIMULATION OR REALDRIVING
        self.COLLECT_DATA = True
        self.RANDOM_DIST_IDX  = None      # Random distance between ego vehicle
                                          # to closest traffic light
        self.TRAINING_IMAGE = None
        self.TRAINING_IMAGE_NO = 1
        self.TRAINING_LABEL = []
        self.TRAINING_DIST  = []
        self.TRAINING_ISSTORED = False

        if (self.COLLECT_DATA):
            tstamp = time.localtime()
            self.TRAINING_FILE = 'Training%d%d%d%d%d%d' \
                         %(tstamp.tm_year, tstamp.tm_mon, tstamp.tm_mday, \
                           tstamp.tm_hour, tstamp.tm_min, tstamp.tm_sec)


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        stop_line_positions = self.config['stop_line_positions']
        # Make KD Tree
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.x] for wp in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        cv_img = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        light_wp, state = self.process_traffic_lights()

        if (self.COLLECT_DATA) and not(self.TRAINING_ISSTORED):
            if ( state != -1 ) and \
                            ( self.traffic_ego_dist < 160.0 )  and \
                            ( self.ego_wp_idx < 9000 ) :
                self.CollectTrainingData(cv_img)
            else:
                self.RANDOM_DIST_IDX = []


            if  self.ego_wp_idx >= 9000:
                # store the result
                train_data_dict = {'Xtrain' : self.TRAINING_IMAGE, \
                                   'YLabel' : self.TRAINING_LABEL, \
                                   'EgoTrafficDist' : self.TRAINING_DIST}
                train_data_fid  = open(self.TRAINING_FILE,'wb')

                pickle.dump(train_data_dict, train_data_fid, protocol=2)
                train_data_fid.close()

                self.TRAINING_ISSTORED = True

        if (light_wp != -1) and (self.ego_wp_idx % 10 == 0):

            if self.TRAINING_IMAGE is not None:
                size_of_training_set = self.TRAINING_IMAGE.shape
            else:
                size_of_training_set = (0,0,0,0)

            print('Light state : %d, light wp-idx : %d, ego idx : %d, distance : %5.2f'% \
                  (state, light_wp, self.ego_wp_idx, self.traffic_ego_dist) )

            print('Training set size (%d,%d,%d,%d)'%size_of_training_set)



        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

<<<<<<< HEAD
    def get_closest_waypoint(self, x, y):
=======
    def get_closest_waypoint(self, pose_x, pose_y):
>>>>>>> a0dfde1cadd07fca35b9e603c4b65d834f8ef87e
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
<<<<<<< HEAD
        closest_index = KDTree.query([x, y], 1)[1]
        return closest_index
=======
        #TODO implement

        closest_idx = self.waypoint_tree.query([pose_x, pose_y], 1)[1]
        return closest_idx
>>>>>>> a0dfde1cadd07fca35b9e603c4b65d834f8ef87e

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
<<<<<<< HEAD
        if(not self.has_image):
            self.prev_light_loc = None
            return False
        
        if self.MODE == 'SIMULATION':
            return light.state
        else:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
=======
        if self.RUNNING_MODE == 'SIMULATION':

            return light.state

        else:
            if(not self.has_image):
                self.prev_light_loc = None
                return False

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

>>>>>>> a0dfde1cadd07fca35b9e603c4b65d834f8ef87e
            #Get classification
            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that corresponds to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
<<<<<<< HEAD
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        #TODO find the closest visible traffic light (if one exists)
        
        diff = len(self.waypoints.waypoints)
        for i, light in enumerate(self.lights):
            line_pos = stop_line_positions[i]
            line_closest_wp_idx = self.get_closest_waypoint(line_pos[0], line_pos[1])
            d = line_closest_wp_idx - car_wp_idx
            if d>0 and d<diff:
                diff = d
                closest_light = light
                line_wp_idx = line_closest_wp_idx 
            

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        
=======
        if (self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            self.ego_wp_idx = car_wp_idx

            # Find closest visible traffic light (if one exist)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if (closest_light):
            state = self.get_light_state(closest_light)
            self.traffic_ego_dist = self.EgoTrafficLightDist(line_wp_idx, car_wp_idx)
            self.light_state = state
            return line_wp_idx, state

        # Otherwise, set light-state to be -1
        self.light_state = -1

>>>>>>> a0dfde1cadd07fca35b9e603c4b65d834f8ef87e
        return -1, TrafficLight.UNKNOWN


    def EgoTrafficLightDist(self, traffic_light_idx, ego_idx):
        x_traffic = self.waypoints_2d[traffic_light_idx][0]
        y_traffic = self.waypoints_2d[traffic_light_idx][1]

        x_ego     = self.waypoints_2d[ego_idx][0]
        y_ego     = self.waypoints_2d[ego_idx][1]

        return math.sqrt((x_traffic - x_ego) * (x_traffic - x_ego) + \
                         (y_traffic - y_ego) * (y_traffic - y_ego))


    def CollectTrainingData(self, img):
        if not self.RANDOM_DIST_IDX:
            # Generate random distance between 0 to 200
            curr_dist =  int(self.traffic_ego_dist)

            if curr_dist == 0:
                # Don't do anything if current distance is zero
                return


            # Compute sample-size
            if curr_dist < 100:
                sample_size = int(min(5, curr_dist))
            else:
                sample_size = 10

            print('Sampling between %d to %d, with sample size %d'%(1,curr_dist,sample_size))

            # Sampling data from range of number between 1 to curr_dist
            self.RANDOM_DIST_IDX = random.sample(range(0,curr_dist),sample_size)
            # sort in descending order
            self.RANDOM_DIST_IDX.sort(reverse=True)

        # Store image only when distance between ego to traffic light is
        # less than first index on self.RANDOM_DIST_IDX
        if self.traffic_ego_dist < self.RANDOM_DIST_IDX[0]:

            # Get original image size
            img_size = img.shape

            if self.TRAINING_IMAGE is not None:
                self.TRAINING_IMAGE = np.append(self.TRAINING_IMAGE, \
                                                img.reshape(1, img_size[0], img_size[1], img_size[2]), \
                                                axis=0)
            else:
                self.TRAINING_IMAGE = img.reshape(1, img_size[0], img_size[1], img_size[2])

            self.TRAINING_LABEL.append(self.light_state)
            self.TRAINING_DIST.append(self.traffic_ego_dist)

            # Remove first-element from self.RANDOM_DIST_IDX
            self.RANDOM_DIST_IDX.remove(self.RANDOM_DIST_IDX[0])

            print('Collect image no : %d'%(self.TRAINING_IMAGE_NO,))
            self.TRAINING_IMAGE_NO = self.TRAINING_IMAGE_NO + 1



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
