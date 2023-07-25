import math
import numpy as np

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return (qx, qy, qz, qw)


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    # Compute the cosine of the angle between the two vectors.
    dot = np.dot(q1, q2)

    # If the dot product is negative, the quaternions
    # have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close for comfort, linearly interpolate
        # and normalize the result.
        result = q1 + t * (q2 - q1)
        return result / np.sqrt(np.dot(result, result))

    # Since dot is in range [0, DOT_THRESHOLD], acos is safe
    theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * t       # theta = angle between v0 and result

    q2_ = q2 - q1 * dot
    q2_ = q2_ / np.sqrt(np.dot(q2_, q2_))  # { v0, v2 } is now an orthonormal basis

    return np.cos(theta) * q1 + np.sin(theta) * q2_

def unpack_pose_xyz_euler(pose):
    return (pose.position.x, pose.position.y, pose.position.z), euler_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
