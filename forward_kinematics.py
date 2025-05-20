import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """
  c = math.cos(angle)
  s = math.sin(angle)
  t = 1.0 - c
  x, y, z = axis

  # matrix_two = np.matmul(np.expand_dims(axis,1), np.expand_dims(axis,1).T)
  # matrix_3 = [[0, -1 * axis[2], axis[1]], [axis[2], 0, -1 * axis[0]], [-1 * axis[1], axis[0], 0]]

  rot_mat = np.array([[t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
                        [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
                        [t * x * z - s * y, t * y * z + s * x, t * z**2 + c]])  
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array
  """
  # take result from above function, append 2 dims
  rot_mat = rotation_matrix(axis, angle)

  T = np.block([[rot_mat, np.expand_dims(v_A, 1)],
                      [np.zeros((1, 3)), 1]])  
  return T

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """

  hip_frame = homogenous_transformation_matrix([0, 0, 1], joint_angles[0], [0, 0, 0])  
  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """

  shoulder_frame = np.matmul(fk_hip(joint_angles), homogenous_transformation_matrix([0, 1, 0], joint_angles[1], [0, -1 * HIP_OFFSET, 0]))
  return shoulder_frame

def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the elbow frame in the base frame
  """

  elbow_wrt_shoulder = homogenous_transformation_matrix([0, 1, 0], joint_angles[2], [0, 0, UPPER_LEG_OFFSET])
  elbow_frame = np.matmul(fk_shoulder(joint_angles), elbow_wrt_shoulder)
  return elbow_frame

def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """

  # remove these lines when you write your solution
  foot_wrt_elbow = homogenous_transformation_matrix([0, 0, 0], 0, [0, 0, LOWER_LEG_OFFSET])
  end_effector_frame = np.matmul(fk_elbow(joint_angles), foot_wrt_elbow)
  return end_effector_frame
