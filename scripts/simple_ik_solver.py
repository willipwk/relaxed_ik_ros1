#! /usr/bin/env python3
import numpy as np
import os
import sys
import yaml
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from relaxed_ik_ros1.scripts.math_utils import get_quaternion_from_euler, euler_from_quaternion, slerp, unpack_pose_xyz_euler
from relaxed_ik_ros1.scripts.math_utils import Pose7d as Pose
# # make python find the package
# sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + '/../')
from relaxed_ik_ros1.scripts.relaxed_ik_rust_demo import RelaxedIKDemo as ik_solver
from relaxed_ik_ros1.scripts.robot import Robot
from relaxed_ik_ros1.scripts.robot_utils import movo_jointangles_fik2rik, movo_jointangles_rik2fik
from typing import List, Tuple

path_to_src = "/home/willipwk/Projects/movo_stack/movo_stack/client/relaxed_ik_ros1/relaxed_ik_core"

class TraceALine:
    def __init__(self):
        print("No tolerances are given, using zero tolerances")
        tolerances = [0, 0, 0, 0, 0, 0]
        
        self.tolerances = []
        self.time_between = 2.0   # time between two keypoints
        self.num_per_goal = 50    # number of intermediate points between two keypoints
        self.start_from_init_pose = True

        assert len(tolerances) % 6 == 0, "The number of tolerances should be a multiple of 6"
        for i in range(int(len(tolerances) / 6)):
            self.tolerances.append([tolerances[i*6],    tolerances[i*6+1], tolerances[i*6+2], 
                                    tolerances[i*6+3],  tolerances[i*6+4], tolerances[i*6+5]])

        deault_setting_file_path = path_to_src + '/configs/settings.yaml'

        setting_file_path = ''
        if setting_file_path == '':
            setting_file_path = deault_setting_file_path

        os.chdir(path_to_src)
        # Load the infomation
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)

        self.robot = Robot(setting_file_path, use_ros=False, path_to_src=path_to_src)
        self.chains_def = settings['chains_def']

        self.ik_solver = ik_solver(path_to_src)

        self.trajectory = None
        self.starting_ee_poses = None
        self.weight_updates = None
        
    
    def rebuild_ik_solver(self, config: List[float]) -> List[float]:
        """
        Rebuild the ik solver, make sure the result is consistent.
        """
        self.ik_solver = ik_solver(path_to_src)
        config = movo_jointangles_fik2rik(config, gripper_value=0.9)
        self.ik_solver.reset(config)
        return config
    

    def set_next_trajectory(self, cur_config: List[float], target_left_pos: np.ndarray, target_right_pos: np.ndarray):
        trajs = []
        starting_config_translated = self.translate_config(cur_config, self.chains_def)
        starting_ee_poses = self.robot.fk(starting_config_translated)
        print("starting ee pose:", starting_ee_poses)
        self.starting_ee_poses = starting_ee_poses
        p0, p1 = unpack_pose_xyz_euler(starting_ee_poses[0]), unpack_pose_xyz_euler(starting_ee_poses[1])
        init_position    = [p0[0], p1[0]]
        init_orientation = [p0[1], p1[1]]
        init_left_pos = np.array(init_position[1] + init_orientation[1])
        init_right_pos = np.array(init_position[0] + init_orientation[0])
        if target_right_pos is not None:
            traj_right = np.vstack([
                init_right_pos,
                target_right_pos,
            ])
            trajs.append(traj_right + np.array([0, 0, 0.015, 0, 0, 0]) + np.array([0, 0, 0.05, 0, 0, 0]))
        else:
            traj_right = None
        if target_left_pos is not None:
            traj_left = np.vstack([
                init_left_pos,
                target_left_pos,
            ])
            trajs.append(traj_left + np.array([0, 0, 0.015, 0, 0, 0]) + np.array([0, 0, 0.05, 0, 0, 0]))
        else:
            traj_left = None
        
        # fill trajectory with initial position if provided trajs are less than num of arms
        shape0 = (trajs[0].shape[0], 1)
        if traj_left is None:
            trajs.append(np.tile(np.array(init_position[1] + init_orientation[1]), shape0))
        elif traj_right is None:
            trajs.insert(0, np.tile(np.array(init_position[0] + init_orientation[0]), shape0))
        
        trajectory = self.generate_trajectory(trajs, self.num_per_goal)
        self.trajectory = trajectory
        self.weight_updates = self.generate_weight_updates(1)


    def set_grasp_trajectory(self, side: str, cur_config: List[float], start: np.ndarray, dz: float = 0.05):
        trajs = []
        starting_config_translated = self.translate_config(cur_config, self.chains_def)
        starting_ee_poses = self.robot.fk(starting_config_translated)
        print("starting ee pose:", starting_ee_poses)
        self.starting_ee_poses = starting_ee_poses
        p0, p1 = unpack_pose_xyz_euler(starting_ee_poses[0]), unpack_pose_xyz_euler(starting_ee_poses[1])
        init_position    = [p0[0], p1[0]]
        init_orientation = [p0[1], p1[1]]
        if side == "left":
            init_pos = np.array(init_position[1] + init_orientation[1])
        elif side == "right":
            init_pos = np.array(init_position[0] + init_orientation[0])
        else:
            raise ValueError(side)
        
        trajs.append(np.vstack([
            init_pos,
            start + np.array([0,0,dz,0,0,0]),
            start,
        ]) + np.array([0, 0, 0.015, 0, 0, 0]) + np.array([0, 0, 0.05, 0, 0, 0]))
        
        # fill trajectory with initial position if provided trajs are less than num of arms
        shape0 = (trajs[0].shape[0], 1)
        if side == "right":
            trajs.append(np.tile(np.array(init_position[1] + init_orientation[1]), shape0))
        elif side == "left":
            trajs.insert(0, np.tile(np.array(init_position[0] + init_orientation[0]), shape0))

        trajectory = self.generate_trajectory(trajs, self.num_per_goal)
        self.trajectory = trajectory
        self.weight_updates = self.generate_weight_updates(2)


    def set_adjustment_trajectory(self, side: str, cur_config: List[float], start: np.ndarray, end: np.ndarray, dz: float = 0.05):
        trajs = []
        starting_config_translated = self.translate_config(cur_config, self.chains_def)
        starting_ee_poses = self.robot.fk(starting_config_translated)
        print("starting ee pose:", starting_ee_poses)
        self.starting_ee_poses = starting_ee_poses
        p0, p1 = unpack_pose_xyz_euler(starting_ee_poses[0]), unpack_pose_xyz_euler(starting_ee_poses[1])
        init_position    = [p0[0], p1[0]]
        init_orientation = [p0[1], p1[1]] 
        if side == "left":
            init_pos = np.array(init_position[1] + init_orientation[1])
        elif side == "right":
            init_pos = np.array(init_position[0] + init_orientation[0])
        else:
            raise ValueError(side)
    
        trajs.append(np.vstack([
            init_pos,
            start + np.array([0,0,dz,0,0,0]),                      # move to end
            start,
            end
        ]) + np.array([0, 0, 0.015, 0, 0, 0]) + np.array([0, 0, 0.05, 0, 0, 0]))
        
        # fill trajectory with initial position if provided trajs are less than num of arms
        shape0 = (trajs[0].shape[0], 1)
        if side == "right":
            trajs.append(np.tile(np.array(init_position[1] + init_orientation[1]), shape0))
        elif side == "left":
            trajs.insert(0, np.tile(np.array(init_position[0] + init_orientation[0]), shape0))

        trajectory = self.generate_trajectory(trajs, self.num_per_goal)
        self.trajectory = trajectory
        self.weight_updates = self.generate_weight_updates(5)


    def generate_trajectory(self, trajs: List[np.ndarray], num_per_goal: int) -> list:
        assert len(trajs) == self.robot.num_active_chains
        trajectory = []
        
        for i in range(len(trajs[0]) - 1):
            for t in np.linspace(0, 1, num_per_goal):
                poses = self.copy_poses(self.starting_ee_poses)
                for k in range(self.robot.num_active_chains):
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

                trajectory.append(poses)
            
        return trajectory


    def generate_weight_updates(self, num_keypoints: int) -> List[dict]:
        """
        Update weight after the first IK (for more smoothness)
        """
        weight_updates = []

        if num_keypoints == 1:
            i = 0
            num_empty_updates = self.num_per_goal
            if self.start_from_init_pose and i == 0:
                upd = {
                    'eepos' : 500.0,
                    'eequat' : 5.0,
                    'minvel'  : 0.01,
                    'minacc'  : 0.01,
                    'minjerk' : 0.01,
                    'selfcollision' : 0.01,
                    'selfcollision_ee' : 0.6,
                    'maxmanip' : 3.0,
                }
                for arm_idx in range(self.robot.num_chain):
                    upd[f"envcollision_{arm_idx}"] = 1.0
            
                weight_updates.append(upd)
                num_empty_updates -= 1
            for _ in range(num_empty_updates):
                weight_updates.append({})
        else:
            for i in range(num_keypoints - 1):
                num_empty_updates = self.num_per_goal
                if self.start_from_init_pose and i == 0:
                    upd = {
                        'eepos' : 500.0,
                        'eequat' : 5.0,
                        'minvel'  : 0.,
                        'minacc'  : 0.,
                        'minjerk' : 0.,
                        'selfcollision' : 0.01,
                        'selfcollision_ee' : 0.6,
                        'maxmanip' : 3.0,
                    }
                    for arm_idx in range(self.robot.num_chain):
                        upd[f"envcollision_{arm_idx}"] = 1.0
                
                    weight_updates.append(upd)
                    num_empty_updates -= 1
                elif self.start_from_init_pose and i == 1:
                    upd = {
                        'eequat' : 5.0,
                        'minvel'  : 0.,
                        'minacc'  : 0.,
                        'minjerk' : 0.,
                        'selfcollision_ee' : 0.6,
                        'jointlimit' : 3.0,
                    }
                    for arm_idx in range(self.robot.num_chain):
                        if self.robot.is_active_chain[arm_idx]:
                            upd[f"envcollision_{arm_idx}"] = 1.0
                        else:
                            upd[f"envcollision_{arm_idx}"] = 1.0
                    weight_updates.append(upd)
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

    
    def get_ik_list_from_traj(self, update_weights: bool = True) -> Tuple[list, float]:
        """
        Generate list of IK solutions from trajectory without the need of ROS.
        """
        assert self.ik_solver is not None, "IK Solver not initialized."
        ik_solutions = []
        target_distance = 0
        for j in range(len(self.trajectory)):
            positions = []
            orientations = []
            tolerances = []
            for i in range(self.robot.num_active_chains):
                positions.extend(self.trajectory[j][i].position.tolist())
                orientations.extend(self.trajectory[j][i].orientation.tolist())
                if i < len(self.tolerances):
                    tolerances.extend(self.tolerances[i])
                else:
                    tolerances.extend(self.tolerances[0])
            if update_weights:
                self.ik_solver.update_objective_weights(self.weight_updates[j])
            
            # print(positions, orientations, tolerances)
            ik_solution = self.ik_solver.solve_pose_goals(positions, orientations, tolerances)
            ik_translated = self.translate_config(ik_solution, self.chains_def)
            ee_pos = self.robot.fk(ik_translated)
            p0, p1 = unpack_pose_xyz_euler(ee_pos[0]), unpack_pose_xyz_euler(ee_pos[1])
            ik_positions = np.array(list(p0[0]) + list(p1[0]))
            gt_positions = np.array(positions)
            if j == len(self.trajectory) - 1:
                target_distance = np.linalg.norm(gt_positions - ik_positions)
            print("distance from ground truth trajectory:", np.linalg.norm(gt_positions - ik_positions))
            print("x y z differences:", ik_positions - gt_positions)
            ik_solutions.append(ik_solution)
            losses = self.ik_solver.query_loss(ik_solution)
            
            print(f"Total Loss: {sum(losses)}")
            # print(self.get_individual_loss(losses, ['eepos_x','eepos_y','eepos_z', "eequat_x","eequat_y", "eequat_z"]))
            print(j)
        
        movo_order_ik_solutions = [movo_jointangles_rik2fik(x) for x in ik_solutions]    
        return movo_order_ik_solutions, target_distance
    
    def translate_config(self, joint_angles, chains_def):
        """
        Handle cases where there are duplicate articulated joints in different chains
        """
        ja_out = []
        for chain in chains_def:
            for joint_idx in chain:
                ja_out.append(joint_angles[joint_idx])
                
        return ja_out
    
    def get_individual_loss(self, losses, query):
        assert self.ik_solver is not None
        ret = {}
        for k in query:
            vals = []
            for i, name in enumerate(self.ik_solver.weight_names):
                if k == name:
                    vals.append(losses[i])
            ret[k] = vals
        return ret
                    
                
def save_ik_list(ik_list, path):
    ik_arr = [movo_jointangles_rik2fik(x) for x in ik_list]
    ik_arr = np.array(ik_arr)
    np.save(path, ik_arr)


if __name__ == '__main__':
    fpath = os.path.dirname(os.path.abspath(__file__))

    trace_a_line = TraceALine()
    
    # Adjust for residual policy
    
    movo_config = np.load(f"{fpath}/ik_seq1_left_0_-1.npy")
    init_movo_config = movo_config[49]
    traj = np.load("../my_traj_files/traject_6d_1_left0.npy")
    rot = traj[0, 3:]

    side = input("Next adjust the left / right hand? (left/right): ")
    cur_rangedik_config = trace_a_line.rebuild_ik_solver(init_movo_config)
    
    start = np.array([0.5698309, 0.14579591, 0.47491103])
    start = np.append(start, rot)
    trace_a_line.set_delta_trajectory(side, cur_rangedik_config, start)
    ik_list_1 = trace_a_line.get_ik_list_from_traj()
    # save_ik_list(ik_list_1, f"{fpath}/ik_seq_debug.npy")