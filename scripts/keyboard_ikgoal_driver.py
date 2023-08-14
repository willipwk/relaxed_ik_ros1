#! /usr/bin/env python3

import readchar
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped, Vector3Stamped, QuaternionStamped, Pose, Twist
from std_msgs.msg import Bool
from relaxed_ik_ros1.msg import EEPoseGoals, EEVelGoals, IKUpdateWeight
import transformations as T
from robot import Robot
from pynput import keyboard
import yaml
from geometry_msgs.msg import Pose, Twist, Vector3
import numpy as np
from math_utils import unpack_pose_xyz_euler

path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'

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
 
  return [qx, qy, qz, qw]

class KeyboardInput:
    def __init__(self):
        deault_setting_file_path = path_to_src + '/configs/settings.yaml'

        setting_file_path = rospy.get_param('setting_file_path')
        if setting_file_path == '':
            setting_file_path = deault_setting_file_path
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)
        self.robot = Robot(setting_file_path)
        self.chains_def = settings['chains_def']
        starting_config_translated = self.translate_config(settings['starting_config'], self.chains_def)
        # self.ee_poses =  self.robot.fk(settings['starting_config'])
        self.starting_ee_poses = self.robot.fk(starting_config_translated)

        self.ee_pose_pub = rospy.Publisher('relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)
        self.ik_weight_pub = rospy.Publisher('relaxed_ik/ik_update_weight', IKUpdateWeight, queue_size=128)
        
        self.pos_stride = 0.005
        self.rot_stride = 0.010

        self.seq = 1
        self.poses = self.copy_poses(self.starting_ee_poses)
        
        # # two arms
        # self.position = [[0.8,-0.5,0.8],[0.8,0.5,0.8]] 
        # self.orientation = [[0,0,0],[0,0,0]]
        
        
        # two arms
        print(self.starting_ee_poses)
        p0, p1 = unpack_pose_xyz_euler(self.starting_ee_poses[0]), unpack_pose_xyz_euler(self.starting_ee_poses[1])
        # self.position = [[0.8,-0.5,0.8],[0.8,0.5,0.8]] 
        # self.orientation = [[0,0,0],[0,0,0]]
        self.position    = [list(p0[0]), list(p1[0])]
        self.orientation = [list(p0[1]), list(p1[1])]
        # print(self.position,self.orientation)
        
        
        try:
            tolerances = rospy.get_param('~tolerances')
        except:
            print("No tolerances are given, using zero tolerances")
            tolerances = [0, 0, 0, 0, 0, 0]
        self.tolerances = []
        assert len(tolerances) % 6 == 0, "The number of tolerances should be a multiple of 6"
        for i in range(int(len(tolerances) / 6)):
            self.tolerances.append(Twist(   Vector3(tolerances[i*6],    tolerances[i*6+1], tolerances[i*6+2]), 
                                            Vector3(tolerances[i*6+3],  tolerances[i*6+4], tolerances[i*6+5])))
        
        keyboard_listener = keyboard.Listener(
            on_press = self.on_press,
            on_release = self.on_release)

        rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        print("starting listener")
        keyboard_listener.start()
        
        # reduce weights of envcollision on figers. for testing purposes
        weight_update = {f'envcollision_{arm_idx}' : 1.0 for arm_idx in [1,2,3,5,6,7]}
        for k, v in weight_update.items():
            msg = IKUpdateWeight()
            msg.weight_name = k
            msg.value = v
            self.ik_weight_pub.publish(msg)
        

    def on_press(self, key):

        if key.char == 'w':
            self.position[0][0] += self.pos_stride
        elif key.char == 'x':
            self.position[0][0] -= self.pos_stride
        elif key.char == 'a':
            self.position[0][1] += self.pos_stride
        elif key.char == 'd':
            self.position[0][1] -= self.pos_stride
        elif key.char == 'q':
            self.position[0][2] += self.pos_stride
        elif key.char == 'z':
            self.position[0][2] -= self.pos_stride
        elif key.char == '1':
            self.orientation[0][0] += self.rot_stride
        elif key.char == '2':
            self.orientation[0][0] -= self.rot_stride
        elif key.char == '3':
            self.orientation[0][1] += self.rot_stride
        elif key.char == '4':
            self.orientation[0][1] -= self.rot_stride
        elif key.char == '5':
            self.orientation[0][2] += self.rot_stride
        elif key.char == '6':
            self.orientation[0][2] -= self.rot_stride
        elif key.char == 'i':
            self.position[1][0] += self.pos_stride
        elif key.char == 'm':
            self.position[1][0] -= self.pos_stride
        elif key.char == 'j':
            self.position[1][1] += self.pos_stride
        elif key.char == 'l':
            self.position[1][1] -= self.pos_stride
        elif key.char == 'u':
            self.position[1][2] += self.pos_stride
        elif key.char == 'n':
            self.position[1][2] -= self.pos_stride    
        elif key.char == '=':
            self.orientation[1][0] += self.rot_stride
        elif key.char == '-':
            self.orientation[1][0] -= self.rot_stride
        elif key.char == '0':
            self.orientation[1][1] += self.rot_stride
        elif key.char == '9':
            self.orientation[1][1] -= self.rot_stride
        elif key.char == '8':
            self.orientation[1][2] += self.rot_stride
        elif key.char == '7':
            self.orientation[1][2] -= self.rot_stride    
            
        elif key.char == ']':
            rospy.signal_shutdown()
        
        print("Position: {}, Rotation: {}".format(self.position, self.orientation))

    def on_release(self, key):
        pass
        #self.position = [0,0,0]
        #self.orientation = [0,0,0]

    def timer_callback(self, event):
        msg = EEPoseGoals()

        for i in range(self.robot.num_active_chains):
            poses = self.copy_poses(self.starting_ee_poses)
            poses[i].position.x = self.position[i][0]
            poses[i].position.y = self.position[i][1]
            poses[i].position.z = self.position[i][2]
            
            quat = get_quaternion_from_euler(self.orientation[i][0], self.orientation[i][1], self.orientation[i][2])
            poses[i].orientation.x = quat[0]
            poses[i].orientation.y = quat[1]
            poses[i].orientation.z = quat[2]
            poses[i].orientation.w = quat[3]
            msg.ee_poses.append(poses[i])
            
            if i < len(self.tolerances):
                msg.tolerances.append(self.tolerances[i])
            else:
                msg.tolerances.append(self.tolerances[0])

        self.seq += 1
        self.ee_pose_pub.publish(msg)
    
    
    
    
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
    rospy.init_node('keyboard_input')
    print("init node complete!")
    keyboard = KeyboardInput()
    rospy.spin()