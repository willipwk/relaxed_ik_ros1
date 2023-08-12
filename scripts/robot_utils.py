import numpy as np

def movo_jointangles_fik2rik(joint_angles, gripper_value):
    """
    translate fastik jointangles (right arm - 7, left arm - 7, linear_joint - 1) 
    to rangedik (1 linear + 7 right arm + 3 fingers + 7 left arm + 3 fingers) 
    """
    assert len(joint_angles) == 7 + 7 + 1
    linear_joint = joint_angles[-1]
    return [linear_joint] + list(joint_angles[:7]) + [gripper_value] * 3 + list(joint_angles[7:14]) + [gripper_value] * 3

def movo_jointangles_rik2fik(joint_angles):
    """
    translate rangedik (1 linear + 7 right arm + 3 fingers + 7 left arm + 3 fingers) 
    to  fastik (right arm - 7, left arm - 7, linear_joint - 1) 
    """
    assert len(joint_angles) == 1 + 7 + 3 + 7 + 3
    return  list(joint_angles[1:8]) + list(joint_angles[11:18]) + [joint_angles[0]]