import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import sensor_msgs
import struct
import std_msgs
from gazebo_msgs.msg import ModelStates
import g2o

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
        self.flag_keypoint_coordinates_available_ = False

        # set params
        self.feature_match_distance_threshold_ = 30
        self.max_depth_measurable_ = 200 # should not be hard-coded
        self.img_width_ = 0
        self.img_height_ = 0
        self.transformation_from_camera_to_world_ = np.array([[ 0,  0, 1,   0],
                                                              [-1,  0, 0,   0],
                                                              [ 0, -1, 0, 1.7],
                                                              [ 0,  0, 0,   1]])

        # set graph params
        self.optimizer_ = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer_.set_algorithm(algorithm)
        self.kernel_ = g2o.RobustKernelTukey() # decide which kernel to use and its threshold
        self.kernel_.set_delta(5)

        # set topic names
        self.rgb_img_topic_   = "/depth_camera/image_raw"
        self.depth_img_topic_ = "/depth_camera/depth/image_raw"
        self.pointcloud_topic_ = "/depth_camera/points"
        self.imu_topic_ = "/imu_ros_plugin/out"
        self.gt_topic_ = "/gazebo/model_states"

    def initSubcribes(self):
        depth_img_sub = Subscriber(self, sensor_msgs.msg.Image, self.depth_img_topic_)
        rgb_img_sub = Subscriber(self, sensor_msgs.msg.Image, self.rgb_img_topic_)
        pointcloud_sub = Subscriber(self, sensor_msgs.msg.PointCloud2, self.pointcloud_topic_)
        self.create_subscription(sensor_msgs.msg.Imu, self.imu_topic_, self.decodeIMU, 1)
        self.create_subscription(ModelStates, self.gt_topic_, self.decodeGT, 1)

        sync = ApproximateTimeSynchronizer([rgb_img_sub, depth_img_sub, pointcloud_sub], queue_size=1, slop=0.05)
        sync.registerCallback(self.decodeImg)

    def initPublishers(self):
        pass
    
    def decodeIMU(self, imu_msg):
        self.linear_acc_ = imu_msg.linear_acceleration
        self.orientation_quat_ = imu_msg.orientation
        self.angular_vel_ = imu_msg.angular_velocity
        # print(self.flag_img_available_)
        self.runSLAM()

    def decodeGT(self, msg):
        rover_index = msg.name.index('rover')
        gt_rover_pose = msg.pose[rover_index]
        position = gt_rover_pose.position
        self.gt_rover_position_ = np.array([position.x, position.y, position.z])
        orientation = gt_rover_pose.orientation
        self.gt_rover_orientation_ = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

    def decodeImg(self, rgb_img_msg, depth_img_msg, pointcloud_msg):
        if not rgb_img_msg.data or not depth_img_msg.data:
            self.flag_img_available_ = False

        else:
            # convert img msgs to np instances
            bridge = CvBridge()
            self.current_rgb_img_   = bridge.imgmsg_to_cv2(rgb_img_msg, 'bgr8')
            self.current_depth_img_ = bridge.imgmsg_to_cv2(depth_img_msg, '32FC1')
            self.flag_img_available_ = True
            self.img_height_ = np.shape(self.current_rgb_img_)[0]
            self.img_width_ = np.shape(self.current_rgb_img_)[1]
            self.decodePointCloud(pointcloud_msg)

            '''
            # TO VISUALIZE THE IMGS
            print(self.current_rgb_img_.shape)
            cv.imshow("rgb", self.current_rgb_img_)
            depth_img_normalized = cv.normalize(self.current_depth_img_, None, 0, 255, cv.NORM_MINMAX)
            depth_img_normalized = depth_img_normalized.astype(np.uint8)
            cv.imshow("depth", depth_img_normalized)
            cv.waitKey(1)
            '''

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
        if (not self.flag_img_available_):
            self.runOdometry()
        
        else:
            if (not self.flag_initial_img_available_):
                self.processInitialFrame()
                return
            
            self.extractFeatures()
            self.matchFeatures()
            if (not self.flag_matches_available_): return
            self.getFeatureCoordinatesIn3D()
            if (not self.flag_keypoint_coordinates_available_): return
            # print(self.prev_feature_coordinates_[0:2]) # coord of first 2 features of prev
            # print(self.prev_good_keypoints_[0].pt) # x,y pixel of the 1st keypoint location for prev
            # print(self.current_feature_coordinates_[0:2]) # coord of first 2 features of current
            # print(self.current_good_keypoints_[0].pt) # x,y pixel of the 1st keypoint location for current
            self.runOdometry()
            self.runICP() # visiual-inertial odometry

            self.flag_img_available_   = False
            # self.prev_descriptors_ = self.current_descriptors_
            self.prev_rgb_img_   = self.current_rgb_img_.copy()
            self.prev_pointcloud_ = self.current_pointcloud_.copy()
            # self.prev_good_depths_ = self.current_good_depths_ ####################################
        
    def processInitialFrame(self):
        self.prev_rgb_img_   = self.current_rgb_img_.copy()
        self.prev_depth_img_ = self.current_depth_img_.copy()
        self.prev_pointcloud_ = self.current_pointcloud_.copy()
        self.flag_initial_img_available_ = True

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

        '''
        # TO VISUALIZE ALL THE MATCHING FEAUTRES NOT ONLY THE GOOD ONES
        img_match = cv.drawMatches(self.prev_rgb_img_, self.prev_keypoints_, self.current_rgb_img_, self.current_keypoints_, matches, None)
        cv.imshow("All Matches", img_match)
        cv.waitKey(1)
        '''

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

        """
        for ind in range(len(self.prev_feature_coordinates_)):
            dist = np.linalg.norm(self.prev_feature_coordinates_[ind] - self.current_feature_coordinates_[ind])
            if dist >= 3: indices_to_remove.add(ind)
        """
        
        for ind in sorted(indices_to_remove, reverse=True):
            del self.prev_feature_coordinates_[ind]
            del self.current_feature_coordinates_[ind]

        if len(self.current_feature_coordinates_) > 0: self.flag_keypoint_coordinates_available_ = True

        """
        print(self.current_feature_coordinates_[:5])
        print("\n")
        print(self.prev_feature_coordinates_[:5])
        print("\n")
        """

        # [x,y,z] = self.current_feature_coordinates_ in the world frame not in camera
        # for the camera frame: origin at the center, z outside the camera, x positive towards right (along width), y positive towards down (along height)

    def runICP(self):
        number_of_vertices = len(self.optimizer_.vertices())
        number_of_edges = len(self.optimizer_.edges())
        
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
        if number_of_edges == 0:
            pose_1 = g2o.VertexSE3()
            pose_1.set_id(number_of_vertices)
            pose_1.set_estimate(g2o.Isometry3d(np.eye(4)))
            self.optimizer_.add_vertex(pose_1)
        
        else:
            pose_1 = self.optimizer_.vertex(self.last_node_id_)
        
        last_transformation_matrix = pose_1.estimate().matrix()

        pose_2 = g2o.VertexSE3()
        pose_2.set_id(number_of_vertices + 1)
        pose_2.set_estimate(g2o.Isometry3d(last_transformation_matrix))
        self.optimizer_.add_vertex(pose_2)
        self.last_node_id_ = number_of_vertices + 1

        # add landmarks
        landmark_list = []
        for ind, lm_pose in enumerate(self.prev_feature_coordinates_):
            landmark = g2o.VertexSE3()
            landmark.set_id(number_of_vertices + 2 + ind)
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
        self.optimizer_.optimize(10)

        optimized_pose2 = pose_2.estimate()
        optimized_pose1 = pose_1.estimate()
        print("Optimized Pose:")
        print(f"gt: {self.gt_rover_position_}")
        print(f"estimated: {optimized_pose2.translation()}")
        print(f"error: {self.gt_rover_position_ - optimized_pose2.translation()}")
        # print(optimized_pose2.matrix())
        print("\n")
        ################################################################

    
    def runOdometry(self):
        """
        number_of_vertices = len(self.optimizer_.vertices())
        number_of_edges = len(self.optimizer_.edges())

        # add increamnetal poses
        if number_of_edges == 0:
            pose_1 = g2o.VertexSE3()
            pose_1.set_id(number_of_vertices)
            pose_1.set_estimate(g2o.Isometry3d(np.eye(4)))
            self.optimizer_.add_vertex(pose_1)
        
        else:
            pose_1 = self.optimizer_.vertex(self.last_node_id_)
        
        last_transformation_matrix = pose_1.estimate().matrix()
        pose_2 = g2o.VertexSE3()
        pose_2.set_id(number_of_vertices + 1)
        pose_2.set_estimate(g2o.Isometry3d(last_transformation_matrix))
        self.optimizer_.add_vertex(pose_2)
        self.last_node_id_ = number_of_vertices + 1

        edge = g2o.EdgeSE3()
        edge.set_id(number_of_edges)
        edge.set_vertex(0, pose_1)
        edge.set_vertex(1, pose_2)
        edge.set_measurement(g2o.Isometry3d(np.eye(3), lm))
        edge.set_information(info_matrix)
        edge.set_robust_kernel(self.kernel_)
        self.optimizer_.add_edge(edge)

        # Optimize
        self.optimizer_.initialize_optimization()
        # self.optimizer_.set_verbose(True)
        self.optimizer_.optimize(10)

        optimized_pose2 = pose_2.estimate()
        optimized_pose1 = pose_1.estimate()
        print("Optimized Pose:")
        print(optimized_pose1.translation())
        print(optimized_pose2.translation())
        print(optimized_pose2.matrix())
        print("\n")
        """
        pass

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