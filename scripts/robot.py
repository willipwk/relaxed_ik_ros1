#! /usr/bin/env python3

import numpy as np
import os
# import rospkg
# import rospy
import transformations as T

# from geometry_msgs.msg import Pose, Vector3
# from std_msgs.msg import Float32MultiArray, Bool, String

from timeit import default_timer as timer
from urdf_parser_py.urdf import URDF
import PyKDL
from kdl_parser import kdl_tree_from_urdf_model
import yaml

class Robot():
    def __init__(self, setting_path=None, path_to_src=None, use_ros=True):
        if use_ros:
            import rospkg
            path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'
        elif path_to_src is not None:
            pass
        else:
            raise ValueError("path_to_src is not passed, and ROS is not installed so project path not known.")
        self.use_ros = use_ros
        
        setting_file_path = path_to_src + '/configs/settings.yaml'
        if setting_path != '':
           setting_file_path = setting_path
        os.chdir(path_to_src)

        # Load the infomation
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)

        urdf_name = settings["urdf"]
        
        self.robot = URDF.from_xml_file(path_to_src + "/configs/urdfs/" + urdf_name)
        self.kdl_tree = kdl_tree_from_urdf_model(self.robot)
        
        # all non-fixed joint         
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_vel_limits = []
        self.all_joint_names = []
        for j in self.robot.joints:
            if j.type != 'fixed':
                self.joint_lower_limits.append(j.limit.lower)
                self.joint_upper_limits.append(j.limit.upper)
                self.joint_vel_limits.append(j.limit.velocity)
                self.all_joint_names.append(j.name)

        # joints solved by relaxed ik
        self.articulated_joint_names = []
        assert len(settings['base_links']) == len(settings['ee_links']) 
        self.num_chain = len(settings['base_links'])
        self.is_active_chain = settings['is_active_chain']
        self.num_active_chains = sum(map(lambda x: 1 if x else 0, settings['is_active_chain']))

        for i in range(self.num_chain):
            arm_chain = self.kdl_tree.getChain( settings['base_links'][i],
                                                settings['ee_links'][i])
            for j in range(arm_chain.getNrOfSegments()):
                joint = arm_chain.getSegment(j).getJoint()
                # 8 is fixed joint   
                if joint.getType() != 8:
                    joint_name = joint.getName()
                    if joint_name not in self.articulated_joint_names:
                        self.articulated_joint_names.append(joint_name)

        if 'starting_config' in settings:
            print(self.articulated_joint_names, settings['starting_config'])
            assert len(self.articulated_joint_names) == len(settings['starting_config']), \
                        "Number of joints parsed from urdf should be the same with the starting config in the setting file."

        self.arm_chains = []
        self.fk_p_kdls = []
        self.fk_v_kdls = []
        self.ik_p_kdls = []
        self.ik_v_kdls = []
        self.num_jnts = []
        for i in range(self.num_chain):
            arm_chain = self.kdl_tree.getChain( settings['base_links'][i],
                                                settings['ee_links'][i])
            self.arm_chains.append(arm_chain)
            self.fk_p_kdls.append(PyKDL.ChainFkSolverPos_recursive(arm_chain))
            self.fk_v_kdls.append(PyKDL.ChainFkSolverVel_recursive(arm_chain))
            self.ik_v_kdls.append(PyKDL.ChainIkSolverVel_pinv(arm_chain))
            self.ik_p_kdls.append(PyKDL.ChainIkSolverPos_NR(arm_chain, 
                                                            self.fk_p_kdls[i],
                                                            self.ik_v_kdls[i]))
            self.num_jnts.append(arm_chain.getNrOfJoints())

    def fk_single_chain(self, fk_p_kdl, joint_angles, num_jnts):
        if self.use_ros:
            from geometry_msgs.msg import Pose
        else:
            from math_utils import Pose7d as Pose
        assert len(joint_angles) == num_jnts, "length of input: {}, number of joints: {}".format(len(joint_angles), num_jnts)
    
        kdl_array = PyKDL.JntArray(num_jnts)
        for idx in range(num_jnts):
            kdl_array[idx] = joint_angles[idx]

        end_frame = PyKDL.Frame()
        fk_p_kdl.JntToCart(kdl_array, end_frame)

        pos = end_frame.p
        rot = PyKDL.Rotation(end_frame.M)
        rot = rot.GetQuaternion()
        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        pose.orientation.x = rot[0]
        pose.orientation.y = rot[1]
        pose.orientation.z = rot[2]
        pose.orientation.w = rot[3]
        return pose

    def fk(self, joint_angles):
        l = 0
        r = 0
        poses = []
        for i in range(self.num_chain):
            r += self.num_jnts[i]
            ja = joint_angles[l:r]
            pose = self.fk_single_chain(self.fk_p_kdls[i], ja, self.num_jnts[i])
            l = r
            if self.is_active_chain[i]:
                poses.append(pose)
        
        return poses

    def fk_single_chain_all_frames(self, chain, all_joint_angles):
        # assert len(joint_angles) == num_jnts, "length of input: {}, number of joints: {}".format(len(joint_angles), num_jnts)
    
        from geometry_msgs.msg import Pose
        joint_frames = []
    
        sub_chain = PyKDL.Chain()

        for i in range(chain.getNrOfSegments()):

            sub_chain.addSegment(chain.getSegment(i))

            fk_solver = PyKDL.ChainFkSolverPos_recursive(sub_chain)
            joint_angles = PyKDL.JntArray(sub_chain.getNrOfJoints())


            for j in range(sub_chain.getNrOfJoints()):
                joint_angles[j] = all_joint_angles[j]


            frame = PyKDL.Frame()

   
            fk_solver.JntToCart(joint_angles, frame)

            pos = frame.p
            rot = PyKDL.Rotation(frame.M)
            rot = rot.GetQuaternion()
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = pos[2]
            pose.orientation.x = rot[0]
            pose.orientation.y = rot[1]
            pose.orientation.z = rot[2]
            pose.orientation.w = rot[3]
            joint_frames.append(pose)

        return joint_frames

    def fk_all_frames(self, joint_angles):
        l = 0
        r = 0
        poses = []
        for i in range(self.num_chain):
            r += self.num_jnts[i]
            pose = self.fk_single_chain_all_frames(self.arm_chains[i], joint_angles[l:r])
            l = r
            poses.append(pose)
        return poses
    
    def get_joint_state_msg(self, joint_angles):
        from sensor_msgs.msg import JointState
        from rospy import Time as rospyTime
        js = JointState()
        js.header.stamp = rospyTime.now()
        js.name = self._joint_names
        js.position = joint_angles
        return js