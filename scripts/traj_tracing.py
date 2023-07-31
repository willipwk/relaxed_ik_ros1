#! /usr/bin/env python3
import math
import csv
import ctypes
import numpy as np
import os
import rospkg
import rospy
import sys
import transformations as T
import yaml

from timeit import default_timer as timer
from geometry_msgs.msg import Pose, Twist, Vector3
from relaxed_ik_ros1.msg import EEPoseGoals, EEVelGoals, IKUpdateWeight
from relaxed_ik_ros1.srv import IKPoseRequest,  IKPose
from robot import Robot
from math_utils import get_quaternion_from_euler, euler_from_quaternion, slerp, unpack_pose_xyz_euler


path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'


class TraceALine:
    def __init__(self):
        try:
            tolerances = rospy.get_param('~tolerances')
        except:
            print("No tolerances are given, using zero tolerances")
            tolerances = [0, 0, 0, 0, 0, 0]

        
        try:
            self.use_topic_not_service = rospy.get_param('~use_topic_not_service')
            print("use_topic_not_service")
        except:
            self.use_topic_not_service = False

        try: 
            self.loop = rospy.get_param('~loop')
        except:
            self.loop = False
        
        self.tolerances = []
        self.time_between = 2.0   # time between two keypoints
        self.num_per_goal = 50    # number of intermediate points between two keypoints
        self.start_from_init_pose = True
        traj_offset = np.array([0,0,0.0,0,0,0])

        assert len(tolerances) % 6 == 0, "The number of tolerances should be a multiple of 6"
        for i in range(int(len(tolerances) / 6)):
            self.tolerances.append(Twist(   Vector3(tolerances[i*6],    tolerances[i*6+1], tolerances[i*6+2]), 
                                            Vector3(tolerances[i*6+3],  tolerances[i*6+4], tolerances[i*6+5])))

        deault_setting_file_path = path_to_src + '/configs/settings.yaml'

        setting_file_path = rospy.get_param('setting_file_path')

        if setting_file_path == '':
            setting_file_path = deault_setting_file_path

        os.chdir(path_to_src)
        # Load the infomation
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)
        
        urdf_file = open(path_to_src + '/configs/urdfs/' + settings["urdf"], 'r')
        urdf_string = urdf_file.read()
        rospy.set_param('robot_description', urdf_string)

        self.robot = Robot(setting_file_path)
        self.chains_def = settings['chains_def']
        starting_config_translated = self.translate_config(settings['starting_config'], self.chains_def)
        # self.ee_poses =  self.robot.fk(settings['starting_config'])
        self.starting_ee_poses = self.robot.fk(starting_config_translated)
        # print(self.starting_ee_poses)
        
        trajs = []
        for traj_file in settings["traj_files"]:
            trajs.append(np.load(path_to_src + '/traj_files/' + traj_file) + traj_offset)

        # print(trajs[0].shape, trajs[1].shape)
        
        
        ### set initial positions
        # self.init_position = [[0.8,-0.5,0.8],[0.8,0.5,0.8]]
        # self.init_orientation = [[0,0,0],[0,0,0]]
        p0, p1 = unpack_pose_xyz_euler(self.starting_ee_poses[0]), unpack_pose_xyz_euler(self.starting_ee_poses[1])
        self.init_position    = [p0[0], p1[0]]
        self.init_orientation = [p0[1], p1[1]] 
        
        trajs_with_init = []
        traj_lengths = []
        
        if self.start_from_init_pose:
            for i, traj in enumerate(trajs):
                init_pos = np.array(self.init_position[i] + self.init_orientation[i])
                trajs_with_init.append(np.vstack([init_pos, traj[0], traj]))
                traj_lengths.append(len(trajs_with_init[i]))
        else:
            trajs_with_init = trajs
            for i, traj in enumerate(trajs):
                traj_lengths.append(len(trajs[i]))
        print(trajs_with_init)
        
        # fill trajectory with initial position if provided trajs are less than num of arms
        if len(trajs_with_init) < self.robot.num_chain:
            shape0 = trajs_with_init[0].shape
            for i in range(len(trajs_with_init), self.robot.num_chain):
                trajs_with_init.append(np.tile(np.array(self.init_position[i] + self.init_orientation[i]), shape0))
                traj_lengths.append(len(trajs_with_init[i]))
        
        assert(all(l == traj_lengths[0] for l in traj_lengths))
        
        self.num_keypoints = traj_lengths[0]
        
        
        
        
        self.trajectory = self.generate_trajectory(trajs_with_init, self.num_per_goal)
        self.weight_updates = self.generate_weight_updates(self.num_keypoints, self.num_per_goal)
        
        print(len(self.trajectory),len(self.weight_updates))
        assert(len(self.trajectory) == len(self.weight_updates))
        
        
        if self.use_topic_not_service:
            self.ee_pose_pub = rospy.Publisher('relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)
        else:
            rospy.wait_for_service('relaxed_ik/solve_pose')
            self.ik_pose_service = rospy.ServiceProxy('relaxed_ik/solve_pose', IKPose)
        self.ik_weight_pub = rospy.Publisher('relaxed_ik/ik_update_weight', IKUpdateWeight, queue_size=128)
        
        
        count_down_rate = rospy.Rate(1)
        count_down = 3
        while not rospy.is_shutdown():
            print("Start line tracing in {} seconds".format(count_down))
            count_down -= 1
            if count_down == 0:
                break
            count_down_rate.sleep()

        self.trajectory_index = 0
        self.timer = rospy.Timer(rospy.Duration(self.time_between / self.num_per_goal), self.timer_callback)

    def generate_trajectory(self, trajs, num_per_goal):
        
        assert len(trajs) == self.robot.num_chain
        trajectory = []
        
        
        for i in range(len(trajs[0]) - 1):
            for t in np.linspace(0, 1, num_per_goal):
                poses = self.copy_poses(self.starting_ee_poses)
                for k in range(self.robot.num_chain):
                    traj = trajs[k]
                    # linear interpolation
                    position_goal = (1 - t) *np.array(traj[i][:3]) + t * np.array(traj[i+1][:3]) 
                    orientation_goal = slerp(np.array(get_quaternion_from_euler(traj[i][3], traj[i][4], traj[i][5])),
                                             np.array(get_quaternion_from_euler(traj[i+1][3], traj[i+1][4], traj[i+1][5])),
                                             t)
                    ( poses[k].position.x, 
                      poses[k].position.y, 
                      poses[k].position.z )    = tuple(position_goal)
                    
                    ( poses[k].orientation.x, 
                      poses[k].orientation.y,
                      poses[k].orientation.z,
                      poses[k].orientation.w ) = tuple(orientation_goal)

                    # for k in range(1, self.robot.num_chain):
                    #     poses[k].position.x = self.init_position[1][0]
                    #     poses[k].position.y = self.init_position[1][1]
                    #     poses[k].position.z = self.init_position[1][2]

                    #     ( poses[k].orientation.x, 
                    #       poses[k].orientation.y,
                    #       poses[k].orientation.z,
                    #       poses[k].orientation.w ) = get_quaternion_from_euler(*self.init_orientation[1])
                trajectory.append(poses)
            
        return trajectory

    def generate_weight_updates(self, num_keypoints, num_per_goal):
        """
        Update weight after the first IK (for more smoothness)
        """
        weight_updates = []
        
        for i in range(num_keypoints - 1):
            num_empty_updates = num_per_goal
            if self.start_from_init_pose and i == 0:
                weight_updates.append({
                    'eepos' : 50.0,
                    'eequat' : 0.0,
                    'minvel'  : 0.5,
                    'minacc'  : 0.3,
                    'minjerk' : 0.1,
                    'selfcollision' : 0.01,
                    'selfcollision_ee' : 0.05,
                    'envcollision': 0.5,
                    'maxmanip' : 3.0,
                })
                num_empty_updates -= 1
            elif self.start_from_init_pose and i == 1:
                weight_updates.append({
                    'eequat'  : 3.0,
                    'minvel'  : 0.7,
                    'minacc'  : 0.5,
                    'minjerk' : 0.3,
                    'selfcollision_ee' : 0.5,
                    'envcollision': 10.0,
                    'jointlimit' : 3.0,
                })
                num_empty_updates -= 1
            for _ in range(num_empty_updates):
                weight_updates.append({})
        
        return weight_updates
                
    
    def copy_poses(self, input_poses):
        output_poses = []
        for i in range(len(input_poses)):
            output_poses.append(self.copy_pose(input_poses[i]))
        return output_poses
    
    def copy_pose(self, input_pose):
        output_pose = Pose()
        output_pose.position.x = input_pose.position.x
        output_pose.position.y = input_pose.position.y
        output_pose.position.z = input_pose.position.z
        output_pose.orientation.x = input_pose.orientation.x
        output_pose.orientation.y = input_pose.orientation.y
        output_pose.orientation.z = input_pose.orientation.z
        output_pose.orientation.w = input_pose.orientation.w
        return output_pose

    def timer_callback(self, event):
        if self.trajectory_index >= len(self.trajectory):
            if self.loop:
                print("Trajectory finished, looping")
                self.trajectory_index = 0
            else:
                rospy.signal_shutdown("Trajectory finished")
            return

        if self.use_topic_not_service:
            ee_pose_goals = EEPoseGoals()
            for i in range(self.robot.num_chain):
                ee_pose_goals.ee_poses.append(self.trajectory[self.trajectory_index][i])
                if i < len(self.tolerances):
                    ee_pose_goals.tolerances.append(self.tolerances[i])
                else:
                    ee_pose_goals.tolerances.append(self.tolerances[0])
            self.ee_pose_pub.publish(ee_pose_goals)
            
            weight_update = self.weight_updates[self.trajectory_index]
            for k, v in weight_update.items():
                msg = IKUpdateWeight()
                msg.weight_name = k
                msg.value = v
                self.ik_weight_pub.publish(msg)
        else:
            req = IKPoseRequest()
            for i in range(self.robot.num_chain):
                req.ee_poses.append(self.trajectory[self.trajectory_index][i])
                if i < len(self.tolerances):
                    req.tolerances.append(self.tolerances[i])
                else:
                    req.tolerances.append(self.tolerances[0])
            
            ik_solutions = self.ik_pose_service(req)

        self.trajectory_index += 1
        
    def translate_config(self, joint_angles, chains_def):
        """
        Handle cases where there are duplicate articulated joints in different chains
        """
        ja_out = []
        for chain in chains_def:
            for joint_idx in chain:
                ja_out.append(joint_angles[joint_idx])
                
        return ja_out
    
if __name__ == '__main__':
    rospy.init_node('LineTracing')
    trace_a_line = TraceALine()
    rospy.spin()
