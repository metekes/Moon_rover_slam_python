import numpy as np

def quaternion2rotation_matrix(quaternion):
    x = quaternion[0, 0]
    y = quaternion[1, 0]
    z = quaternion[2, 0]
    w = quaternion[3, 0]

    return np.array([[x**2-y**2-z**2+w**2, 2*(x*y+z*w), 2*(x*z-y*w)],
                     [2*(x*y-z*w), -x**2+y**2-z**2+w**2, 2*(y*z+x*w)],
                     [2*(z*x+y*w), 2*(z*y-x*w), -x**2-y**2+z**2+w**2]]).reshape(3,3)

def quaternion_derivative(angular_velocity, quaternion):
    x = angular_velocity[0, 0]
    y = angular_velocity[1, 0]
    z = angular_velocity[2, 0]

    omega = np.array([[0, z, -y, x],
                      [-z, 0, x, y],
                      [y, -x, 0, z],
                      [-x, -y, -z, 0]]).reshape(4,4)
    
    return 0.5 * omega @ quaternion.reshape(4,1)

def project2Dto3D(cam_intrinsic_mat, pixel_x, pixel_y, depth):
    return np.linalg.inv(cam_intrinsic_mat) @ (np.array([pixel_x, pixel_y, 1]) * depth)

def project3Dto2D(cam_intrinsic_mat, x, y, z):
    return cam_intrinsic_mat @ np.array([x/z, y/z, 1.0], dtype=float)