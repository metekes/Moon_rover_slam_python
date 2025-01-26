import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy

import sensor_msgs
import struct
import std_msgs
from gazebo_msgs.msg import ModelStates

import g2o
import cv2 as cv
import numpy as np

from .helper import *
from .customVertexSE3 import *

class GraphSLAM(Node):

    def __init__(self):
        super().__init__('rover_slam_python_node')
        self.initClassVariables()
        self.initSubcribes()
        self.initPublishers()

    def initClassVariables(self):
        # set flags
        self.flag_initial_img_available_ = False
        self.flag_img_available_ = False
        self.flag_initial_imu_data_available_ = False
        self.flag_imu_data_available_ = False
        self.flag_keypoint_coordinates_available_ = False
        self.flag_gt_available = False ########################################################### sil

        # set params
        self.frame_count_from_keyframe_ = 0
        self.keyframe_list_ = []
        self.feature_match_distance_threshold_ = 30 # should be tuned
        self.max_feature_3D_distance_ = 7 # should be tuned
        self.num_closest_feature_cooridantes_ = 15 # should be tuned
        self.max_depth_measurable_ = 8 # should be tuned
        self.img_width_ = 0
        self.img_height_ = 0
        self.transformation_correction_camera_tilt_ = np.array([[ 1,  0, 0, 0],
                                                                [ 0,  np.cos(0), np.sin(0), 0],
                                                                [ 0, -np.sin(0), np.cos(0), 0],
                                                                [ 0,  0, 0, 1]])
        
        self.transformation_from_camera_to_world_ = self.transformation_correction_camera_tilt_ @ np.array([[ 0,  0, 1,   0.215],
                                                                                                            [-1,  0, 0,   0],
                                                                                                            [ 0, -1, 0, 0.065],
                                                                                                            [ 0,  0, 0,   1]])
        self.gravity_ = np.array([0, 0, -1.62]).reshape(3,1)
        # self.matrix_A_ = np.array([]) # for filter implementation
        # self.matrix_B_ = np.array([]) # for filter implementation
        self.velocity_ = np.zeros((3,1))
        self.position_ = np.zeros((3,1))
        self.orientation_quat_= np.array([0, 0, 0, 1], dtype=float).reshape(4,1) # from origin to rover, transpose of pose.estimate().matrix()[:3,:3]
        self.count_ = 0 ######################################################################################################################################## sil

        # set graph params
        self.optimizer_ = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer_.set_algorithm(algorithm)
        self.kernel_ = g2o.RobustKernelTukey() # decide which kernel to use and its threshold
        self.kernel_.set_delta(5)
        self.rover_vertex_ids_ = set()

        # set topic names
        self.rgb_img_topic_   = "/camera/image_raw"
        self.depth_img_topic_ = "/camera/depth/image_raw"
        self.pointcloud_topic_ = "/camera/points"
        self.imu_topic_ = "/imu"
        self.gt_topic_ = "/model_states"

        # store gt and est positions to plot them later
        self.file_ = open("positions.txt", "w")
        self.file_.write("GT_x, GT_y, GT_z, Est_x, Est_y, Est_z\n")

    def initSubcribes(self):
        qos_profile = QoSProfile(
            depth=10, 
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        depth_img_sub = Subscriber(self, sensor_msgs.msg.Image, self.depth_img_topic_, qos_profile=qos_profile)
        rgb_img_sub = Subscriber(self, sensor_msgs.msg.Image, self.rgb_img_topic_, qos_profile=qos_profile)
        pointcloud_sub = Subscriber(self, sensor_msgs.msg.PointCloud2, self.pointcloud_topic_, qos_profile=qos_profile)
        self.create_subscription(sensor_msgs.msg.Imu, self.imu_topic_, self.decodeIMU, qos_profile=qos_profile)
        self.create_subscription(ModelStates, self.gt_topic_, self.decodeGT, qos_profile=qos_profile)

        sync = ApproximateTimeSynchronizer([rgb_img_sub, depth_img_sub, pointcloud_sub], queue_size=1, slop=0.05)
        sync.registerCallback(self.decodeImg)

    def initPublishers(self):
        pass
    
    def decodeIMU(self, imu_msg):
        self.flag_imu_data_available_ = True
        linear_acc = imu_msg.linear_acceleration
        self.current_linear_acc_ = np.array([linear_acc.x, linear_acc.y, linear_acc.z]).reshape(3,1)
        angular_vel = imu_msg.angular_velocity
        self.current_angular_vel_ = np.array([angular_vel.x, angular_vel.y, angular_vel.z]).reshape(3,1)
        self.current_imu_time_ = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 10**-9
        self.runSLAM()

    def decodeGT(self, msg):
        rover_index = msg.name.index('rover')

        gt_rover_pose = msg.pose[rover_index]
        self.gt_rover_position_ = np.array([gt_rover_pose.position.x, gt_rover_pose.position.y, gt_rover_pose.position.z])
        self.gt_rover_orientation_ = np.array([gt_rover_pose.orientation.x, gt_rover_pose.orientation.y, gt_rover_pose.orientation.z, gt_rover_pose.orientation.w])

        twist = msg.twist[rover_index]
        self.gt_rover_vel_ = np.array([twist.linear.x, twist.linear.y, twist.linear.z]).reshape(3,1)

        ####################################################################################
        if not self.flag_gt_available:
            self.flag_gt_available = True
            self.velocity_ = self.gt_rover_vel_.copy()
            self.position_ = self.gt_rover_position_.reshape(3,1).copy()
            self.orientation_quat_= self.gt_rover_orientation_.reshape(4,1).copy()
        ######################################################################################

    def decodeImg(self, rgb_img_msg, depth_img_msg, pointcloud_msg):
        if not rgb_img_msg.data or not depth_img_msg.data:
            self.flag_img_available_ = False

        else:
            # convert img msgs to np instances
            bridge = CvBridge()
            self.current_rgb_img_   = bridge.imgmsg_to_cv2(rgb_img_msg, 'bgr8')
            self.current_depth_img_ = bridge.imgmsg_to_cv2(depth_img_msg, '32FC1')
            self.current_depth_img_ = np.nan_to_num(self.current_depth_img_, nan=0.0, posinf=0.0, neginf=0.0)
            self.flag_img_available_ = True
            self.img_height_ = np.shape(self.current_rgb_img_)[0]
            self.img_width_ = np.shape(self.current_rgb_img_)[1]
            self.current_img_time_ = rgb_img_msg.header.stamp.sec + rgb_img_msg.header.stamp.nanosec * 10**-9
            path = "/home/mete/rover_slam_python/photos/" + str(self.count_) + ".jpg" 
            cv.imwrite(path, self.current_rgb_img_)
            self.decodePointCloud(pointcloud_msg)
            self.frame_count_from_keyframe_ += 1

            """
            # TO VISUALIZE THE IMGS
            print(self.current_rgb_img_.shape)
            cv.imshow("rgb", self.current_rgb_img_)
            self.current_depth_img_ = np.nan_to_num(self.current_depth_img_, nan=0.0, posinf=0.0, neginf=0.0)
            depth_img_normalized = cv.normalize(self.current_depth_img_, None, 0, 255, cv.NORM_MINMAX)
            depth_img_normalized = depth_img_normalized.astype(np.uint8)
            cv.imshow("depth", depth_img_normalized)
            cv.waitKey(1)
            """
            

    def decodePointCloud(self, pointcloud_msg):
        point_step = pointcloud_msg.point_step
        data = pointcloud_msg.data
        num_points = len(data) // point_step
        pointcloud_x = []
        pointcloud_y = []
        pointcloud_z = []

        for i in range(num_points):
            start = i * point_step
            # Decode x, y, z using the offsets
            x = struct.unpack_from('f', data, start + 0)[0]
            y = struct.unpack_from('f', data, start + 4)[0]
            z = struct.unpack_from('f', data, start + 8)[0]
            pointcloud_x.append(x)
            pointcloud_y.append(y)
            pointcloud_z.append(z)

        pointcloud_x = (np.array(pointcloud_x)).reshape(self.img_height_, self.img_width_)
        pointcloud_y = (np.array(pointcloud_y)).reshape(self.img_height_, self.img_width_)
        pointcloud_z = (np.array(pointcloud_z)).reshape(self.img_height_, self.img_width_)
        self.current_pointcloud_ = np.array([pointcloud_x, pointcloud_y, pointcloud_z])

    def runSLAM(self):
        # self.flag_imu_data_available_ = False # to run ICP only
        if not self.flag_gt_available: return
        if (not self.flag_initial_imu_data_available_):
            self.resetIMUData()
        
        elif ((not self.flag_img_available_) and self.flag_imu_data_available_):
            self.runOdometry()
            self.resetIMUData()

        elif ((not self.flag_imu_data_available_) and self.flag_img_available_):
            if (not self.flag_initial_img_available_):
                orb = cv.ORB_create()
                _, kf_descriptors = orb.detectAndCompute(cv.cvtColor(self.current_rgb_img_, cv.COLOR_BGR2GRAY), None)
                self.keyframe_list_.append(kf_descriptors.copy())
                self.resetImgData()
                return
            
            self.extractFeatures()
            self.matchFeatures()
            if (not self.flag_matches_available_): return
            self.getFeatureCoordinatesIn3D()
            if (not self.flag_keypoint_coordinates_available_): return
            self.runICP() # visiual-inertial odometry

            self.resetImgData()
        
        elif (self.flag_imu_data_available_ and self.flag_img_available_):
            self.runOdometry()
            self.resetIMUData()

            if (not self.flag_initial_img_available_):
                orb = cv.ORB_create()
                _, kf_descriptors = orb.detectAndCompute(cv.cvtColor(self.current_rgb_img_, cv.COLOR_BGR2GRAY), None)
                self.keyframe_list_.append(kf_descriptors.copy())
                self.resetImgData()
                return
            self.extractFeatures()
            self.matchFeatures()
            if (not self.flag_matches_available_): return
            self.getFeatureCoordinatesIn3D()
            if (not self.flag_keypoint_coordinates_available_): return

            self.runICP() # visiual-inertial odometry

            self.resetImgData()
            # self.prev_descriptors_ = self.current_descriptors_
            # self.prev_good_depths_ = self.current_good_depths_ ####################################
    
    def resetIMUData(self):
        self.prev_angular_vel_ = self.current_angular_vel_.copy()
        self.prev_linear_acc_ = self.current_linear_acc_.copy()
        self.prev_imu_time_ = self.current_imu_time_
        self.flag_initial_imu_data_available_ = True
        self.flag_imu_data_available_ = False

    def resetImgData(self):
        self.prev_rgb_img_   = self.current_rgb_img_.copy()
        self.prev_depth_img_ = self.current_depth_img_.copy()
        self.prev_pointcloud_ = self.current_pointcloud_.copy()
        self.prev_img_time_= self.current_img_time_
        self.flag_initial_img_available_ = True
        self.flag_img_available_ = False
        self.flag_keypoint_coordinates_available_ = False

    def extractFeatures(self):
        orb = cv.ORB_create()
        self.prev_keypoints_, self.prev_descriptors_       = orb.detectAndCompute(cv.cvtColor(self.prev_rgb_img_, cv.COLOR_BGR2GRAY), None)
        self.current_keypoints_, self.current_descriptors_ = orb.detectAndCompute(cv.cvtColor(self.current_rgb_img_, cv.COLOR_BGR2GRAY), None)

    def matchFeatures(self):
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.prev_descriptors_, self.current_descriptors_)
        good_matches = [m for m in matches if m.distance < self.feature_match_distance_threshold_]
        
        if len(good_matches) == 0:
            self.flag_matches_available_ = False
            return
        
        self.flag_matches_available_ = True
        self.prev_good_keypoints_    = []
        self.current_good_keypoints_ = []

        for match in good_matches:
            self.prev_good_keypoints_.append(self.prev_keypoints_[match.queryIdx])
            self.current_good_keypoints_.append(self.current_keypoints_[match.trainIdx])
        
        self.keyframeSelection()

        """
        # TO VISUALIZE ALL THE MATCHING FEAUTRES NOT ONLY THE GOOD ONES
        img_match = cv.drawMatches(self.prev_rgb_img_, self.prev_keypoints_, self.current_rgb_img_, self.current_keypoints_, matches, None)
        cv.imshow("All Matches", img_match)
        cv.waitKey(1)
        """

    def keyframeSelection(self):
        if self.frame_count_from_keyframe_ == 20: 
            self.keyframe_list_.append(self.current_good_keypoints_.copy())
            self.frame_count_from_keyframe_ = 0
        
        elif len(self.keyframe_list_) != 0:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.keyframe_list_[-1], self.current_descriptors_)
            good_matches = [m for m in matches if m.distance < self.feature_match_distance_threshold_]
            if len(good_matches) < 0.9 * len(self.current_descriptors_):
                self.keyframe_list_.append(self.current_descriptors_.copy())
                self.frame_count_from_keyframe_ = 0

    def getFeatureCoordinatesIn3D(self):
        self.prev_feature_coordinates_ = []
        self.current_feature_coordinates_ = []
        indices_to_remove = set()

        for ind, keypoint in enumerate(self.prev_good_keypoints_):
            x,y = keypoint.pt
            position = self.prev_pointcloud_[:,int(y),int(x)]
            if position[-1] >= self.max_depth_measurable_: indices_to_remove.add(ind)
            position = self.transformation_from_camera_to_world_ @ np.vstack((position.reshape(3,1), np.array([1])))
            self.prev_feature_coordinates_.append(position[:3])

        for ind, keypoint in enumerate(self.current_good_keypoints_):
            x,y = keypoint.pt
            position = self.current_pointcloud_[:,int(y),int(x)]
            if position[-1] >= self.max_depth_measurable_: indices_to_remove.add(ind)
            position = self.transformation_from_camera_to_world_ @ np.vstack((position.reshape(3,1), np.array([1])))
            self.current_feature_coordinates_.append(position[:3])

        # find features to seperate from eachtoher in 3D
        for ind in range(len(self.prev_feature_coordinates_)):
            dist = np.linalg.norm(self.prev_feature_coordinates_[ind] - self.current_feature_coordinates_[ind])
            if dist >= self.max_feature_3D_distance_: indices_to_remove.add(ind)
        
        # remove features further away from maximum measurable distance and the features two seperate from each other in 3D
        for ind in sorted(indices_to_remove, reverse=True):
            del self.prev_feature_coordinates_[ind]
            del self.current_feature_coordinates_[ind]

        # select only the features with smallest difference between 3D coordiantes
        if (len(self.current_feature_coordinates_)) > self.num_closest_feature_cooridantes_:
            # Calculate norms for each pair of features
            norms = [
                (i, np.linalg.norm(self.prev_feature_coordinates_[i] - self.current_feature_coordinates_[i]))
                for i in range(len(self.prev_feature_coordinates_))
            ]

            # Sort by norm value and select the indices of the smallest 5
            smallest_indices = [i for i, _ in sorted(norms, key=lambda x: x[1])[:self.num_closest_feature_cooridantes_]]

            # Filter the coordinates based on the smallest indices
            self.prev_feature_coordinates_ = [self.prev_feature_coordinates_[i] for i in smallest_indices]
            self.current_feature_coordinates_ = [self.current_feature_coordinates_[i] for i in smallest_indices]

        # check if feature coordinates available
        if len(self.current_feature_coordinates_) > 2: self.flag_keypoint_coordinates_available_ = True

        """
        print(self.current_feature_coordinates_[:5])
        print("\n")
        print(self.prev_feature_coordinates_[:5])
        print("\n")
        """

        # [x,y,z] = self.current_feature_coordinates_ in the world frame not in camera
        # for the camera frame: origin at the center, z outside the camera, x positive towards right (along width), y positive towards down (along height)

    def runICP(self):
        
        """
        # To test if the performace degrades when the num of features are less than 6
        count = 0
        for i in range(len(self.prev_feature_coordinates_)):
            dist = self.prev_feature_coordinates_[i] - self.current_feature_coordinates_[i]
            if np.linalg.norm(dist) < 3: count = count + 1
        
        if count < 6: return
        print(count)
        """

        """
        for i in range(len(self.prev_feature_coordinates_)):
            dist = self.prev_feature_coordinates_[i] - self.current_feature_coordinates_[i]
            if np.linalg.norm(dist) < 2:
                print("-------------------------------")
                print(self.prev_feature_coordinates_[i])
                print(self.current_feature_coordinates_[i])
                print("-------------------------------")
        """
        """
        print(len(self.prev_feature_coordinates_))
        if len(self.prev_feature_coordinates_) >= 6:
            print(self.prev_feature_coordinates_[:5])
            print(self.current_feature_coordinates_[:5])
        """
        # add incremental poses
        if (not self.flag_imu_data_available_):
            pose_1, pose_2 = self.setImgPoses()
        
        else:
            pose_1, pose_2 = self.matchIMUandImgTimes()

        number_of_vertices = len(self.optimizer_.vertices())
        number_of_edges = len(self.optimizer_.edges())

        last_transformation_matrix = pose_1.estimate().matrix()
        # add landmarks
        landmark_list = []
        for ind, lm_pose in enumerate(self.prev_feature_coordinates_):
            landmark = g2o.VertexSE3()
            landmark.set_id(number_of_vertices + ind)
            lm_pose = last_transformation_matrix @ np.vstack((lm_pose.reshape(3,1), 1))
            landmark.set_estimate(g2o.Isometry3d(np.eye(3), lm_pose[:3]))
            landmark.set_fixed(True)
            self.optimizer_.add_vertex(landmark)
            landmark_list.append(landmark)

        # add edges
        info_matrix = np.zeros((6,6))
        info_matrix[:3, :3] = np.eye(3)

        # add edges between prev pose and landmarks
        for ind, lm in enumerate(self.prev_feature_coordinates_):
            edge = g2o.EdgeSE3()
            edge.set_id(number_of_edges + ind)
            edge.set_vertex(0, pose_1)
            edge.set_vertex(1, landmark_list[ind])
            edge.set_measurement(g2o.Isometry3d(np.eye(3), lm))
            edge.set_information(info_matrix)
            edge.set_robust_kernel(self.kernel_)
            self.optimizer_.add_edge(edge)

        number_of_edges = len(self.optimizer_.edges())

        # add edges between current pose and landmarks
        for ind, lm in enumerate(self.current_feature_coordinates_):
            edge = g2o.EdgeSE3()
            edge.set_id(number_of_edges + ind)
            edge.set_vertex(0, pose_2)
            edge.set_vertex(1, landmark_list[ind])
            edge.set_measurement(g2o.Isometry3d(np.eye(3), lm))
            edge.set_information(info_matrix)
            edge.set_robust_kernel(self.kernel_)
            self.optimizer_.add_edge(edge)

        # Optimize
        self.optimizer_.initialize_optimization()
        # self.optimizer_.set_verbose(True)
        self.optimizer_.optimize(20)
        pose_1.set_fixed((True))
        pose_2.set_fixed((True))

        # store optimized poses
        optimized_pose2 = pose_2.estimate()
        self.file_.write(f"{self.gt_rover_position_[0]}, {self.gt_rover_position_[1]}, {self.gt_rover_position_[2]}, {optimized_pose2.translation()[0]}, {optimized_pose2.translation()[1]}, {optimized_pose2.translation()[2]}\n")

        print("Optimized Pose:")
        print("ICP")
        print(f"gt: {self.gt_rover_position_}")
        print(f"estimated: {optimized_pose2.translation()}")
        print(f"error: {self.gt_rover_position_ - optimized_pose2.translation()}")
        print(f"error:\n {quaternion2rotation_matrix(self.gt_rover_orientation_.reshape(4,1)) @ optimized_pose2.matrix()[:3,:3]}")
        # print(optimized_pose2.matrix())
        print("\n")
    
    def setImgPoses(self):
        number_of_vertices = len(self.optimizer_.vertices())
        if number_of_vertices == 0:
            pose_1 = VertexSE3WithTime()
            pose_1.set_id(number_of_vertices)
            pose_1.set_estimate(g2o.Isometry3d(quaternion2rotation_matrix(self.gt_rover_orientation_.reshape(4,1)).transpose(), self.gt_rover_position_))
            self.rover_vertex_ids_.add(pose_1.id())
        
        else:
            pose_1 = self.optimizer_.vertex(self.last_node_id_)
        
        last_transformation_matrix = pose_1.estimate().matrix()

        pose_2 = VertexSE3WithTime()
        pose_2.set_id(number_of_vertices + 1)
        pose_2.set_estimate(g2o.Isometry3d(last_transformation_matrix))
        self.optimizer_.add_vertex(pose_2)
        self.last_node_id_ = number_of_vertices + 1
        self.rover_vertex_ids_.add(pose_2.id())
        return pose_1, pose_2

    def matchIMUandImgTimes(self):
        closest_pose_1 = None
        closest_time_diff = float('inf')

        for pose_id in self.rover_vertex_ids_:
            vertex = self.optimizer_.vertex(pose_id)  # Replace `self.graph` with your graph instance

            # Calculate the time difference
            time_diff = abs(vertex.get_timestamp() - self.prev_img_time_*1000)
            
            # Update the closest vertex if this one is closer
            if time_diff < closest_time_diff:
                closest_pose_1 = vertex
                closest_time_diff = time_diff

        closest_pose_2 = None
        closest_time_diff = float('inf')

        for pose_id in self.rover_vertex_ids_:
            vertex = self.optimizer_.vertex(pose_id)  # Replace `self.graph` with your graph instance

            # Calculate the time difference
            time_diff = abs(vertex.get_timestamp() - self.prev_img_time_*1000)
            
            # Update the closest vertex if this one is closer
            if time_diff < closest_time_diff:
                closest_pose_2 = vertex
                closest_time_diff = time_diff
        
        return closest_pose_1, closest_pose_2
    
    def runOdometry(self):
        self.count_ += 1 ############################################ sil
        if not self.flag_gt_available: return
        
        self.propagateFilter()
        number_of_vertices = len(self.optimizer_.vertices())
        number_of_edges = len(self.optimizer_.edges())

        # add incremental poses
        if number_of_vertices == 0:
            pose_1 = VertexSE3WithTime()
            pose_1.set_timestamp(self.prev_imu_time_*1000)
            pose_1.set_id(number_of_vertices)
            pose_1.set_estimate(g2o.Isometry3d(quaternion2rotation_matrix(self.gt_rover_orientation_.reshape(4,1)).transpose(), self.gt_rover_position_))
            pose_1.set_fixed(True)
            self.optimizer_.add_vertex(pose_1)
            self.rover_vertex_ids_.add(pose_1.id())
        
        else:
            pose_1 = self.optimizer_.vertex(self.last_node_id_)
            pose_1.set_timestamp(self.prev_imu_time_*1000)
        
        pose_2 = VertexSE3WithTime()
        pose_2.set_timestamp(self.current_imu_time_*1000)
        pose_2.set_id(number_of_vertices + 1)
        pose_2.set_estimate(g2o.Isometry3d(quaternion2rotation_matrix(self.orientation_quat_).transpose(), self.position_))
        self.optimizer_.add_vertex(pose_2)
        self.last_node_id_ = number_of_vertices + 1

        self.rover_vertex_ids_.add(pose_2.id())

        transformation_pose2_to_pose1 = np.linalg.inv(pose_1.estimate().matrix()) @ pose_2.estimate().matrix()

        edge = g2o.EdgeSE3()
        edge.set_id(number_of_edges)
        edge.set_vertex(0, pose_1)
        edge.set_vertex(1, pose_2)
        edge.set_measurement(g2o.Isometry3d(transformation_pose2_to_pose1))
        info_matrix = np.eye(6)*10**-1
        edge.set_information(info_matrix)
        edge.set_robust_kernel(self.kernel_)
        self.optimizer_.add_edge(edge)

        # Optimize
        self.optimizer_.initialize_optimization()
        # self.optimizer_.set_verbose(True)
        self.optimizer_.optimize(10)
        pose_1.set_fixed((True))
        pose_2.set_fixed((True))

        # store poses
        optimized_pose2 = pose_2.estimate()
        self.file_.write(f"{self.gt_rover_position_[0]}, {self.gt_rover_position_[1]}, {self.gt_rover_position_[2]}, {optimized_pose2.translation()[0]}, {optimized_pose2.translation()[1]}, {optimized_pose2.translation()[2]}\n")

        if self.count_ % 20 == 0:
            print("Optimized Pose:")
            print("IMU")
            print(f"gt: {self.gt_rover_position_}")
            print(f"estimated: {optimized_pose2.translation()}")
            print(f"error: {self.gt_rover_position_ - optimized_pose2.translation()}")
            print(f"error:\n {quaternion2rotation_matrix(self.gt_rover_orientation_.reshape(4,1)) @ optimized_pose2.matrix()[:3,:3]}")
            # print(optimized_pose2.matrix())
            print("\n")
        
    def propagateFilter(self):
        dt = self.current_imu_time_ - self.prev_imu_time_
        linear_acc = (self.prev_linear_acc_ + self.current_linear_acc_) / 2
        angular_vel = (self.prev_angular_vel_ + self.current_angular_vel_) / 2

        acc_inertial = (quaternion2rotation_matrix(self.orientation_quat_).transpose() @ linear_acc + self.gravity_)

        self.position_ += self.velocity_ * dt + 0.5 * acc_inertial * dt**2

        self.velocity_ += acc_inertial * dt

        self.orientation_quat_ += quaternion_derivative(angular_vel, self.orientation_quat_) * dt
        self.orientation_quat_ = self.orientation_quat_ / np.linalg.norm(self.orientation_quat_)
        
        """
        ##########################################################################################
        if self.count_ % 100 == 0:
            # print(acc_inertial)
            print(f"gt: {self.gt_rover_position_}")
            print(f"estimated: {self.position_}")
            print(f"error: {self.gt_rover_position_.reshape(3,1) - self.position_}")
            # print(optimized_pose2.matrix())
            print("\n")
        return
        ###########################################################################################
        """

def main(args=None):
    rclpy.init(args=args)

    graphSLAM = GraphSLAM()

    rclpy.spin(graphSLAM)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    graphSLAM.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()