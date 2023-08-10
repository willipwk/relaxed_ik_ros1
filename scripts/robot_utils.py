import numpy as np

def movo_jointangles_fik2rik(joint_angles):
    """
    translate fastik jointangles (right arm - 7, left arm - 7, linear_joint - 1) 
    to rangedik ( linear_joint - 1, right arm - 7, left arm - 7) 
    """
    assert len(joint_angles) == 15
    return [joint_angles[-1]] + list(joint_angles[:-1])

def movo_jointangles_rik2fik(joint_angles):
    """
    translate rangedik ( linear_joint - 1, right arm - 7, left arm - 7) 
    to (right arm - 7, left arm - 7, linear_joint - 1) 
    """
    assert len(joint_angles) == 15
    return  list(joint_angles[1:]) + [joint_angles[0]]