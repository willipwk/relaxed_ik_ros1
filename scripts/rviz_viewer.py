#! /usr/bin/env python3

'''
author: Danny Rakita, Haochen Shi, Yeping Wang
email: rakita@cs.wisc.edu, hshi74@wisc.edu, yeping@cs.wisc.edu
last update: 01/24/23
'''

import numpy
import os
import rospkg
import rospy
import transformations as T
import yaml
import legacy_utils as utils

from robot import Robot
from interactive_markers.interactive_marker_server import *
from sensor_msgs.msg import JointState, PointCloud2, PointField
from visualization_msgs.msg import *
import subprocess
from urdf_parser_py.urdf import URDF
from geometry_msgs.msg import Point
from relaxed_ik_ros1.msg import EEPoseGoals, EEVelGoals, ResetJointAngles

path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'
animation_folder_path = path_to_src + '/animation_files/'
geometry_folder_path = path_to_src + '/geometry_files/'

time_cur = 0.0

def print_cb(msg):
    p = msg.pose.position
    print(msg.marker_name + " is now at [" + str(p.x) + ", " + str(p.y) + ", " + str(p.z) + "]")
    
    
def set_collision_world(server:InteractiveMarkerServer, fixed_frame, env_settings):
    dyn_obs_handles = []

    if 'obstacles' in env_settings:
        obstacles = env_settings['obstacles']
    else:
        raise NameError('Please define the obstacles in the environment!')

    if 'cuboids' in obstacles: 
        cuboids = obstacles['cuboids']
        if cuboids is not None:
            for c in cuboids:
                if c['animation'] == 'static':
                    is_dynamic = 0
                elif c['animation'] == 'interactive':
                    is_dynamic = 1
                else:
                    is_dynamic = 2
                print(c)
                c_quat = T.quaternion_from_euler(c['rotation'][0], c['rotation'][1], c['rotation'][2])
                int_marker = make_marker_env(c['name'], fixed_frame, "cuboid", c['scale'], 
                                        c['translation'], c_quat, is_dynamic)
                server.insert(int_marker, print_cb)
                if is_dynamic == 2:
                    path = animation_folder_path + c['animation']
                    relative_waypoints = utils.read_cartesian_path(path)
                    waypoints = utils.get_abs_waypoints(relative_waypoints, int_marker.pose)
                    dyn_obs_handles.append((int_marker.name, waypoints))

    if 'spheres' in obstacles:
        spheres = obstacles['spheres']
        if spheres is not None:
            for s in spheres:
                if s['animation'] == 'static':
                    is_dynamic = 0
                elif s['animation'] == 'interactive':
                    is_dynamic = 1
                else:
                    is_dynamic = 2
                int_marker = make_marker_env(s['name'], fixed_frame, "sphere", [s['scale']] * 3, 
                                        s['translation'], [1.0,0.0,0.0,0.0], is_dynamic)
                server.insert(int_marker, print_cb)
                if is_dynamic == 2:
                    path = animation_folder_path + s['animation']
                    relative_waypoints = utils.read_cartesian_path(path)
                    waypoints = utils.get_abs_waypoints(relative_waypoints, int_marker.pose)
                    dyn_obs_handles.append((int_marker.name, waypoints))

    if 'point_cloud' in obstacles: 
        point_cloud = obstacles['point_cloud']
        if point_cloud is not None:
            for pc in point_cloud:
                pc_path = geometry_folder_path + pc['file']
                pc_scale = pc['scale']
                pc_points = []
                with open(pc_path, 'r') as point_cloud_file:
                    lines = point_cloud_file.read().split('\n')
                    for line in lines:
                        pt = line.split(' ')
                        if utils.is_point(pt):
                            point = Point()
                            point.x = float(pt[0]) * pc_scale[0]
                            point.y = float(pt[1]) * pc_scale[1]
                            point.z = float(pt[2]) * pc_scale[2]
                            pc_points.append(point)
                
                if pc['animation'] == 'static':
                    is_dynamic = 0
                elif pc['animation'] == 'interactive':
                    is_dynamic = 1
                else:
                    is_dynamic = 2
                pc_quat = T.quaternion_from_euler(pc['rotation'][0], pc['rotation'][1], pc['rotation'][2])
                int_marker = make_marker_env(pc['name'], fixed_frame, "point_cloud", [0.01, 0.01, 0.01], pc['translation'], pc_quat, is_dynamic, points=pc_points)
                server.insert(int_marker, print_cb)
                if is_dynamic == 2:
                    path = animation_folder_path + pc['animation']
                    relative_waypoints = utils.read_cartesian_path(path)
                    waypoints = utils.get_abs_waypoints(relative_waypoints, int_marker.pose)
                    dyn_obs_handles.append((int_marker.name, waypoints))

    server.applyChanges()

    return dyn_obs_handles

class RvizViewer:
    def __init__(self):
        deault_setting_file_path = path_to_src + '/configs/settings.yaml'

        setting_file_path = rospy.get_param('setting_file_path')
        if setting_file_path == '':
            print("Rviz viewer: no setting file path is given, using default setting files --" + setting_file_path)
            setting_file_path = deault_setting_file_path

        os.chdir(path_to_src )

        # Load the infomation
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)
        
        urdf_file = open(path_to_src + '/configs/urdfs/' + settings["urdf"], 'r')
        urdf_string = urdf_file.read()
        rospy.set_param('robot_description', urdf_string)

        self.robot = Robot(setting_file_path)

        subprocess.Popen(["roslaunch", "relaxed_ik_ros1", "rviz_viewer.launch", 
                                                "fixed_frame:=" + settings['base_links'][0]])

        self.js_pub = rospy.Publisher('joint_states',JointState,queue_size=5)
        self.js_msg = JointState()
        self.js_msg.name = self.robot.all_joint_names
        self.js_msg.position = [0] * len(self.robot.all_joint_names)

        # Markers to visualize goal poses
        self.server = InteractiveMarkerServer("simple_marker")

        if 'starting_config' not in settings:
            settings['starting_config'] = [0] * len(self.robot.articulated_joint_names)
        else:
            self.starting_config = settings['starting_config']
        self.chains_def = settings['chains_def']
        starting_config_translated = self.translate_config(settings['starting_config'], self.chains_def)
        print(starting_config_translated)
        # self.ee_poses =  self.robot.fk(settings['starting_config'])
        self.ee_poses = self.robot.fk(starting_config_translated)
        for i in range(self.robot.num_active_chains):
            pose_goal_marker = make_marker('arm_'+str(i), settings['base_links'][i],
                 'widget', [0.1,0.1,0.1], self.ee_poses[i], False)
            self.server.insert(pose_goal_marker)

        # marker_test = make_marker_env('test',settings['base_links'][0], 'cuboid',[0.5,0.5,0.5], [1,1,1], [0,0,0,1], is_dynamic=0)
        # self.server.insert(marker_test)

        frame = self.robot.fk_all_frames(starting_config_translated)
        
        # wait for robot state publisher to start
        rospy.sleep(2.0)
        
        print("all joint names:", self.robot.all_joint_names)
        print("articulated joint names:", self.robot.articulated_joint_names)
        # print(frame[1][-1])
        
        # move the robot in rviz to initial position
        for i in range(len(self.robot.articulated_joint_names)):
            self.js_msg.position[self.robot.all_joint_names.index(self.robot.articulated_joint_names[i])] = \
                                                            settings['starting_config'][i]
        self.js_msg.header.stamp = rospy.Time.now()
        self.js_pub.publish(self.js_msg)
        # ik_reset_pub = rospy.Publisher('/relaxed_ik/reset_ja', ResetJointAngles, queue_size=5)
        
        # js_msg = ResetJointAngles()
        # js_msg.joint_angles = settings['starting_config']
        # rospy.sleep(2.0)
        # ik_reset_pub.publish(js_msg)

        rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointState, self.ja_solution_cb)
        rospy.Subscriber('/relaxed_ik/vis_ee_poses', EEPoseGoals, self.ee_pose_goal_cb)
        rospy.Subscriber('/relaxed_ik/ee_pose_goals', EEPoseGoals, self.ee_pose_goal_cb)
        rospy.Subscriber('/relaxed_ik/ee_vel_goals', EEVelGoals, self.ee_vel_goal_cb)
        
        
        dyn_obs_handles = set_collision_world(self.server, fixed_frame=settings['base_links'][0], env_settings=settings)
        print("set collision world done")
        _ = input()
        
        delta_time = 0.01
        
        while not rospy.is_shutdown():

            if True:
                updated = False
                for (name, waypoints) in dyn_obs_handles:
                    if time_cur < len(waypoints) * delta_time:
                        (time, pose) = utils.linear_interpolate_waypoints(waypoints, int(time_cur / delta_time))
                        self.server.setPose(name, pose)
                        updated = True

                if updated:
                    self.server.applyChanges()

    def ja_solution_cb(self, msg):
        self.js_msg.header.stamp = rospy.Time.now()
        for i in range(len(msg.name)):
            self.js_msg.position[self.robot.all_joint_names.index(msg.name[i])] = msg.position[i]
        self.js_pub.publish(self.js_msg)

    def ee_pose_goal_cb(self, msg):
        assert len(msg.ee_poses) == self.robot.num_active_chains
        for i in range(self.robot.num_active_chains):
            self.ee_poses[i] = msg.ee_poses[i]
        self.update_marker()

    def ee_vel_goal_cb(self, msg):
        assert len(msg.ee_vels) == self.robot.num_active_chains
        for i in range(self.robot.num_active_chains):
            self.ee_poses[i].position.x += msg.ee_vels[i].linear.x
            self.ee_poses[i].position.y += msg.ee_vels[i].linear.y
            self.ee_poses[i].position.z += msg.ee_vels[i].linear.z
            curr_q = [self.ee_poses[i].orientation.w, self.ee_poses[i].orientation.x, self.ee_poses[i].orientation.y, self.ee_poses[i].orientation.z]
            tmp_q = T.quaternion_from_scaledAxis([msg.ee_vels[i].angular.x, msg.ee_vels[i].angular.y, msg.ee_vels[i].angular.z])
            after_q = T.quaternion_multiply(tmp_q, curr_q)
            self.ee_poses[i].orientation.w = after_q[0]
            self.ee_poses[i].orientation.x = after_q[1]
            self.ee_poses[i].orientation.y = after_q[2]
            self.ee_poses[i].orientation.z = after_q[3]
        self.update_marker()

    def update_marker(self):
        for i in range(self.robot.num_active_chains):
            self.server.setPose('arm_'+str(i), self.ee_poses[i])
        self.server.applyChanges()
        
    def translate_config(self, joint_angles, chains_def):
        """
        Handle cases where there are duplicate articulated joints in different chains
        """
        ja_out = []
        for chain in chains_def:
            for joint_idx in chain:
                ja_out.append(joint_angles[joint_idx])
                
        return ja_out

def make_marker(name, fixed_frame, shape, scale, pose, is_dynamic, 
                points=None, color=[0.0,0.5,0.5,1.0], marker_scale=0.3):                
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = fixed_frame
    int_marker.name = name
    int_marker.pose = pose

    int_marker.scale = marker_scale

    origin = Point()
    x_axis = Point()
    x_axis.x = scale[0]
    y_axis = Point()
    y_axis.y = scale[1]
    z_axis = Point()
    z_axis.z = scale[2]
    points = [[origin, x_axis], [origin, y_axis], [origin, z_axis]]
    colors = [[1.0, 0.0, 0.0, 0.6], [0.0, 1.0, 0.0, 0.6], [0.0, 0.0, 1.0, 0.6]]
    for i in range(len(colors)):
        marker = Marker()
        marker.type = Marker.ARROW
        marker.scale.x = 0.01
        marker.scale.y = 0.02
        marker.scale.z = 0.03
        marker.color.r = colors[i][0]
        marker.color.g = colors[i][1]
        marker.color.b = colors[i][2]
        marker.color.a = colors[i][3]
        marker.points = points[i]

        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(marker)
        int_marker.controls.append(control)
  
    return int_marker
def make_marker_env(name, fixed_frame, shape, scale, ts, quat, is_dynamic, 
                points=None, color=[0.0,0.5,0.5,1.0], marker_scale=0.3):                
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = fixed_frame
    int_marker.name = name
    int_marker.pose.position.x = ts[0]
    int_marker.pose.position.y = ts[1]
    int_marker.pose.position.z = ts[2]

    int_marker.pose.orientation.x = quat[1]
    int_marker.pose.orientation.y = quat[2]
    int_marker.pose.orientation.z = quat[3]
    int_marker.pose.orientation.w = quat[0]

    int_marker.scale = marker_scale

    if shape == 'widget':
        origin = Point()
        origin.x = 0.0
        origin.y = 0.0
        origin.z = 0.0
        x_axis = Point()
        x_axis.x = scale[0]
        x_axis.y = 0.0
        x_axis.z = 0.0
        y_axis = Point()
        y_axis.x = 0.0
        y_axis.y = scale[1]
        y_axis.z = 0.0
        z_axis = Point()
        z_axis.x = 0.0
        z_axis.y = 0.0
        z_axis.z = scale[2]
        points = [[origin, x_axis], [origin, z_axis], [origin, y_axis]]
        colors = [[1.0, 0.0, 0.0, 0.6], [0.0, 1.0, 0.0, 0.6], [0.0, 0.0, 1.0, 0.6]]
        for i in range(len(colors)):
            marker = Marker()
            marker.type = Marker.ARROW
            marker.scale.x = 0.01
            marker.scale.y = 0.02
            marker.scale.z = 0.03
            marker.color.r = colors[i][0]
            marker.color.g = colors[i][1]
            marker.color.b = colors[i][2]
            marker.color.a = colors[i][3]
            marker.points = points[i]

            control =  InteractiveMarkerControl()
            control.always_visible = True
            control.markers.append(marker)
            int_marker.controls.append(control)
    else:
        marker = Marker()
        marker.scale.x = scale[0] * 2
        marker.scale.y = scale[1] * 2
        marker.scale.z = scale[2] * 2
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        if shape == "cuboid":
            marker.type = Marker.CUBE
        elif shape == "sphere":
            marker.type = Marker.SPHERE
        elif shape == "point_cloud":
            marker.type = Marker.POINTS
            marker.points = points

        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(marker)
        int_marker.controls.append(control)

    # 1 means that this obstacle is interactive
    if is_dynamic == 1:
        c = 1.0 / numpy.sqrt(2)
        tx_control = InteractiveMarkerControl()
        tx_control.orientation.w = c
        tx_control.orientation.x = c
        tx_control.orientation.y = 0
        tx_control.orientation.z = 0
        tx_control.name = "move_x"
        tx_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(tx_control)

        ty_control = InteractiveMarkerControl()
        ty_control.orientation.w = c
        ty_control.orientation.x = 0
        ty_control.orientation.y = 0
        ty_control.orientation.z = c
        ty_control.name = "move_y"
        ty_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(ty_control)

        tz_control = InteractiveMarkerControl()
        tz_control.orientation.w = c
        tz_control.orientation.x = 0
        tz_control.orientation.y = c
        tz_control.orientation.z = 0
        tz_control.name = "move_z"
        tz_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(tz_control)

        if shape != "sphere":
            rx_control = InteractiveMarkerControl()
            rx_control.orientation.w = c
            rx_control.orientation.x = c
            rx_control.orientation.y = 0
            rx_control.orientation.z = 0
            rx_control.name = "rotate_x"
            rx_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(rx_control)

            ry_control = InteractiveMarkerControl()
            ry_control.orientation.w = c
            ry_control.orientation.x = 0
            ry_control.orientation.y = 0
            ry_control.orientation.z = c
            ry_control.name = "rotate_y"
            ry_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(ry_control)

            rz_control = InteractiveMarkerControl()
            rz_control.orientation.w = c
            rz_control.orientation.x = 0
            rz_control.orientation.y = c
            rz_control.orientation.z = 0
            rz_control.name = "rotate_z"
            rz_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(rz_control)

    return int_marker
if __name__ == '__main__':
    print("RVIZ_VIEWER.PY")
    rospy.init_node('rviz_viewer')
    rviz_viewer = RvizViewer()
    rospy.spin()

