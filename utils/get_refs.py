import pinocchio as pin
from pinocchio.rpy import rpyToMatrix
import numpy as np
# import read_data_kinematics
# import read_data_moments
# import marches
import os
from mat4py import loadmat, savemat

from pinocchio.utils import se3ToXYZQUAT


from utils import marches
from utils import read_data_kinematics
from utils import read_data_moments

def degrees2radians(degrees):
    return degrees * np.pi / 180


def load_kinematics(path, studied_walk):

    data_kinematics = read_data_kinematics.read_data(
        path + "/walk_data/fichiers txt/" + studied_walk + ".txt",
        path + "/walk_data/fichiers txt/" + studied_walk + ".csv",
    )

    return data_kinematics

def load_moments(path, studied_walk):

    data_moment = read_data_moments.read_data_moments(
        path
        + "/walk_data/InverseDynamics/résultats sto & txt/"
        + str(studied_walk)
        + ".txt",
        path
        + "/walk_data/InverseDynamics/résultats sto & txt/"
        + str(studied_walk)
        + ".csv",
    )
    right_moments = [
        data_moment.hip_flexion_r_moment,
        data_moment.knee_angle_r_moment,
        data_moment.ankle_angle_r_moment,
    ]
    left_moments = [
        data_moment.hip_flexion_l_moment,
        data_moment.knee_angle_l_moment,
        data_moment.ankle_angle_l_moment,
    ]

    return(data_moment, left_moments, right_moments)

def create_walk(path, studied_walk, data_kinematics, joints):

    data_moment, left_moments, right_moments = load_moments(path, studied_walk)

    marche = marches.Marche(
        studied_walk,
        data_kinematics.time,
        [joints[31], joints[33], joints[36]],
        [joints[40], joints[42], joints[45]],
        data_moment.time,
        left_moments,
        right_moments,
    )

    return(marche)

def load_markers(path, studied_markers):

    data_markers_global = loadmat(path + "/walk_data/" + studied_markers + ".mat")
    data_markers = data_markers_global["output"]["marker_data"]["Markers"]

    return data_markers

def load_events(path, studied_walk, marche, delay=0):

    path_events = path + "/walk_data/dataEvents/"
    events = loadmat(path_events + str(studied_walk) + "Events.mat")
    marche.toesoff(marche.time, events["Left_Foot_Off"], events["Right_Foot_Off"])
    marche.heelstrike(
        marche.time, events["Left_Foot_Strike"], events["Right_Foot_Strike"]
    )

    t_hs_left = np.array(marche.heelstrike_left) - delay
    t_hs_right = np.array(marche.heelstrike_right) - delay
    t_to_left = np.array(marche.toesoff_left) - delay
    t_to_right = np.array(marche.toesoff_right) - delay

    return (t_hs_left, t_hs_right, t_to_left, t_to_right)


def parse_joints(data_kinematics):

    lumbar_x = degrees2radians(data_kinematics.lumbar_bending)
    lumbar_y = -degrees2radians(data_kinematics.lumbar_extension)
    lumbar_z = degrees2radians(data_kinematics.lumbar_rotation)

    left_hip_AA = -degrees2radians(data_kinematics.hip_adduction_l)
    left_hip_FE = -degrees2radians(data_kinematics.hip_flexion_l)
    left_hip_rot = -degrees2radians(data_kinematics.hip_rotation_l)

    left_knee_FE = -degrees2radians(data_kinematics.knee_angle_l)
    left_knee_rot = degrees2radians(data_kinematics.knee_rot_l)

    left_ankle_AA = degrees2radians(data_kinematics.ankle_add_l)
    left_ankle_FE = -degrees2radians(data_kinematics.ankle_angle_l)
    left_ankle_rot = degrees2radians(data_kinematics.ankle_rot_l)

    left_foot_FE = -degrees2radians(data_kinematics.mtp_angle_l)

    right_hip_AA = degrees2radians(data_kinematics.hip_adduction_r)
    right_hip_FE = -degrees2radians(data_kinematics.hip_flexion_r)
    right_hip_rot = degrees2radians(data_kinematics.hip_rotation_r)

    right_knee_FE = -degrees2radians(data_kinematics.knee_angle_r)
    right_knee_rot = degrees2radians(data_kinematics.knee_rot_r)

    right_ankle_AA = degrees2radians(data_kinematics.ankle_add_r)
    right_ankle_FE = -degrees2radians(data_kinematics.ankle_angle_r)
    right_ankle_rot = degrees2radians(data_kinematics.ankle_rot_r)

    right_foot_FE = -degrees2radians(data_kinematics.mtp_angle_r)

    zeros = np.zeros(len(lumbar_x))

    joints = np.array( [lumbar_x, lumbar_y, zeros, zeros, zeros, zeros, zeros, zeros, lumbar_z, 
                    zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
                    zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, 
                    zeros, zeros, zeros, zeros, zeros, 
                    left_hip_AA, left_hip_FE, left_hip_rot,
                    left_knee_FE, left_knee_rot,
                    left_ankle_AA, left_ankle_FE, left_ankle_rot,
                    left_foot_FE,
                    right_hip_AA, right_hip_FE, right_hip_rot,
                    right_knee_FE, right_knee_rot,
                    right_ankle_AA, right_ankle_FE, right_ankle_rot,
                    right_foot_FE], dtype = np.dtype('d')).T

    return joints

def parse_pelvis(data_kinematics):

    translation_x = data_kinematics.pelvis_tx
    translation_y = -data_kinematics.pelvis_tz
    translation_z = data_kinematics.pelvis_ty

    orientation_x = degrees2radians(data_kinematics.pelvis_list)
    orientation_y = -degrees2radians(data_kinematics.pelvis_tilt)
    orientation_z = degrees2radians(data_kinematics.pelvis_rotation)

    quat_pelvis = np.array(len(translation_x) * [4 * [None]])


    for k in range(len(translation_x)):
        rot_matrix = pin.rpy.rpyToMatrix(orientation_x[k], orientation_y[k], orientation_z[k])
        quat = pin.Quaternion(rot_matrix)
        quat_pelvis[k]= np.array([quat[0], quat[1], quat[2], quat[3]])

    config = np.array([translation_x, translation_y, translation_z,
                       quat_pelvis[:, 0], quat_pelvis[:, 1], quat_pelvis[:,2], quat_pelvis[:,3]]).T
    
    assert len(config[0]) == 7

    return config



def parse_markers(data_markers):

    markers = { m:  1e-3*np.array(l) for m,l in data_markers.items() }
 
    updated_markers = { m: np.array([l[:,1], -l[:,0], l[:,2]]) for m,l in markers.items() }
    print(type(updated_markers))

    return(updated_markers)


def create_q_list(config, joints):

    assert len(config)==len(joints)
    n = len(config)

    q_list = n * [None]

    for i in range(n):
        q_list[i] = np.hstack([config[i], joints[i]])
        assert q_list[i].shape == (7 + 48,)

    return q_list

def save(path, studied_walk, studied_markers, delay=0):

    data_kinematics = load_kinematics(path, studied_walk)

    joints = parse_joints(data_kinematics)
    config = parse_pelvis(data_kinematics)
    q_list = create_q_list(config, joints)

    marche = create_walk(path, studied_walk, data_kinematics, joints)

    data_markers = load_markers(path, studied_markers)
    markers = parse_markers(data_markers)

    t_hs_left, t_hs_right, t_to_left, t_to_right = load_events(path, studied_walk, marche, delay)


    np.savez(
        path + "/new_data/" + studied_walk + "_refs.npz",
        q=q_list,
        t_hs_left=t_hs_left,
        t_hs_right=t_hs_right,
        t_to_left=t_to_left,
        t_to_right=t_to_right,
        time=data_kinematics.time,
        markers=markers
    )

def save_visu_data(path, waist_se3, right_foot_se3, left_foot_se3):
    np.savez(
        path + "/new_data/" + studied_walk + "_refs_from_visu.npz",
        waist_se3=waist_se3,
        right_foot_se3=right_foot_se3,
        left_foot_se3=left_foot_se3
    )

def main():

    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dt = 5e-3

    studied_walk = "marche 1"
    studied_markers = "markers_data"
    delay = 255 * dt

    save(path, studied_walk, studied_markers, delay)

    studied_walk2 = 'marche sans attelle 1'
    studied_markers2 = 'markers_data_patho'
    delay2 = dt

    save(path, studied_walk2, studied_markers2, delay2)
    


 


if __name__ == "__main__":

    main()
