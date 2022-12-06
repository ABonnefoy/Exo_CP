##
## Copyright (c) 2022, LAAS-CNRS ???
##
## Authors: Aurélie Bonnefoy, Sabrina Otmani
##
## This file is part of an ongoing work on the interaction of an exoskeleton and a human suffering prom cerebral palsy
##
## Use: visualise the data acquired from walk experiments for the 
##
##patient with cerebral palsy
##


import numpy as np
import matplotlib.pyplot as plt
import time
import pinocchio as pin
import tsid
import os
from mat4py import loadmat
import math
import plotly.graph_objects as go

from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer

# Used for debug
from IPython import embed
import sys

# Utils
from utils import marches
from utils import read_data_kinematics
from utils import read_data_moments


###### Functions Creation ######


def degrees2radians(degrees):
    return degrees * math.pi / 180


def radians2degrees(radians):
    return radians * 180 / math.pi


####### End of Functions creation ######


# AFFICHAGE
DISPLAY = True
DISPLAY_N = 10
PLOT_JOINTS = True
PLOT_MARKERS = True

# PATH & URD
path = os.path.dirname(os.path.realpath(__file__))
urdf = path + "/urdf/patho_human.urdf"


# Creation du robot TSID
robot_tsid = tsid.RobotWrapper(
    urdf,
    [urdf],
    pin.JointModelFreeFlyer(),
    False,
)

# Creation du robot pinocchio
robot = pin.RobotWrapper.BuildFromURDF(
    urdf, package_dirs=None, root_joint=pin.JointModelFreeFlyer(), verbose=False
)


if DISPLAY:
    robot.initViewer(loadModel=True)
    robot.display(robot.q0)
    gv=robot.viewer.gui
    gv.addFloor("world/floor")


# Configuration
model = robot_tsid.model()
data = robot_tsid.data()
q = np.zeros(robot_tsid.nq)

t = 0.0  # time
dt = 0.005

########################### READ DATA (kinematics, moments, events) ###########################

studied_walk = 'marche sans attelle 1'
studied_markers = 'markers_data_patho'
delay = dt

## Read data
data_kinematics = read_data_kinematics.read_data(
    path + "/walk_data/fichiers txt/" + studied_walk + ".txt",
    path + "/walk_data/fichiers txt/" + studied_walk + ".csv",
)

# Joints positions extracted from data_kinematics
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

translation_x = data_kinematics.pelvis_tx
translation_y = -data_kinematics.pelvis_tz
translation_z = data_kinematics.pelvis_ty

orientation_x = degrees2radians(data_kinematics.pelvis_list)
orientation_y = -degrees2radians(data_kinematics.pelvis_tilt)
orientation_z = degrees2radians(data_kinematics.pelvis_rotation)


# Read Moments
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

# Walk construction
marche = marches.Marche(
    studied_walk,
    data_kinematics.time,
    [left_hip_FE, left_knee_FE, left_ankle_FE],
    [right_hip_FE, right_knee_FE, right_ankle_FE],
    data_moment.time,
    left_moments,
    right_moments,
)


# Read events
path_events = path + "/walk_data/dataEvents/"
events = loadmat(path_events + str(studied_walk) + "Events.mat")
marche.toesoff(data_kinematics.time, events["Left_Foot_Off"], events["Right_Foot_Off"])
marche.heelstrike(
    data_kinematics.time, events["Left_Foot_Strike"], events["Right_Foot_Strike"]
)

# Read FirstFoot
firstFoot = marche.firstFoot()

# Events timestamps
t_hs_left = np.array(marche.heelstrike_left) - delay
t_hs_right = np.array(marche.heelstrike_right) - delay
t_to_left = np.array(marche.toesoff_left) - delay
t_to_right = np.array(marche.toesoff_right) - delay


# Read markers
data_markers_global = loadmat(path + "/walk_data/" + studied_markers + ".mat")
data_markers = data_markers_global["output"]["marker_data"]["Markers"]

markers = { m:  1e-3*np.array(l) for m,l in data_markers.items() }
mnames = list(markers.keys())
T = len(markers[mnames[0]])


## Pairing of markers
pairing = {
    'LASIS': [robot.model.getFrameId('Pelvis')],
    'RASIS': [robot.model.getFrameId('Pelvis')],
    'LPSIS': [robot.model.getFrameId('Pelvis')],
    'RPSIS': [robot.model.getFrameId('Pelvis')],
    'RGT': [robot.model.getFrameId('RightUpperLeg')],
    'RMFE': [robot.model.getFrameId('RightUpperLeg')],
    'RLFE': [robot.model.getFrameId('RightUpperLeg')],
    'LGT': [robot.model.getFrameId('LeftUpperLeg')],
    'LMFE': [robot.model.getFrameId('LeftUpperLeg')],
    'LLFE': [robot.model.getFrameId('LeftUpperLeg')],
    'RATT': [robot.model.getFrameId('RightLowerLeg')],
    'RLM': [robot.model.getFrameId('RightLowerLeg')],
    'RSPH': [robot.model.getFrameId('RightLowerLeg')],
    'LATT': [robot.model.getFrameId('LeftLowerLeg')],
    'LLM': [robot.model.getFrameId('LeftLowerLeg')],
    'LSPH': [robot.model.getFrameId('LeftLowerLeg')],
    'RCAL': [robot.model.getFrameId('RightFoot')],
    'RTT2': [robot.model.getFrameId('RightToe')],
    'RMFH5': [robot.model.getFrameId('RightToe')],
    'RMFH1': [robot.model.getFrameId('RightToe')],
    'LCAL': [robot.model.getFrameId('LeftFoot')],
    'LTT2': [robot.model.getFrameId('LeftToe')],
    'LMFH5': [robot.model.getFrameId('LeftToe')],
    'LMFH1': [robot.model.getFrameId('LeftToe')],
    'SEL': [robot.model.getFrameId('Head')],
    'OCC': [robot.model.getFrameId('Head')],
    'LTEMP': [robot.model.getFrameId('Head')],
    'RTEMP': [robot.model.getFrameId('Head')],
    'SUP': [robot.model.getFrameId('Neck')],
    'C7': [robot.model.getFrameId('Neck')],
    'T10': [robot.model.getFrameId('T12')],
    'STR': [robot.model.getFrameId('T12')],
    'RA': [robot.model.getFrameId('RightUpperArm')],
    'RMHE': [robot.model.getFrameId('RightUpperArm')],
    'RLHE': [robot.model.getFrameId('RightUpperArm')],
    'LA': [robot.model.getFrameId('LeftUpperArm')],
    'LMHE': [robot.model.getFrameId('LeftUpperArm')],
    'LLHE': [robot.model.getFrameId('LeftUpperArm')],
    'RRS': [robot.model.getFrameId('RightForeArm')],
    'RUS': [robot.model.getFrameId('RightForeArm')],
    'LUS': [robot.model.getFrameId('LeftForeArm')],
    'LRS': [robot.model.getFrameId('LeftForeArm')],
    'RHMH2': [robot.model.getFrameId('RightHand')],
    'RFT3': [robot.model.getFrameId('RightHand')],
    'RHMH5': [robot.model.getFrameId('RightHand')],
    'LHMH2': [robot.model.getFrameId('LeftHand')],
    'LHMH5': [robot.model.getFrameId('LeftHand')],
    'LFT3': [robot.model.getFrameId('LeftHand')]}

jointToColor = { robot.model.getFrameId('Pelvis'): [ 50,205,50,1 ],
                 robot.model.getFrameId('RightUpperLeg'): [ 32,178,170,1 ],
                 robot.model.getFrameId('LeftUpperLeg'): [ 32,178,170,1 ],
                 robot.model.getFrameId('RightLowerLeg'): [ 0,206,209,1 ],
                 robot.model.getFrameId('LeftLowerLeg'): [ 0,206,209,1 ],
                 robot.model.getFrameId('RightFoot'): [ 0,0,205,1 ],
                 robot.model.getFrameId('RightToe'): [ 153,50,204,1 ],
                 robot.model.getFrameId('LeftFoot'): [ 0,0,205,1 ],
                 robot.model.getFrameId('LeftToe'): [ 153,50,204,1 ],
                 robot.model.getFrameId('Head'): [ 255,215,0,1 ],
                 robot.model.getFrameId('Neck'): [ 255,255,0,1 ],
                 robot.model.getFrameId('T12'): [ 0,100,0,1 ],
                 robot.model.getFrameId('RightUpperArm'): [ 255,140,0,1 ],
                 robot.model.getFrameId('LeftUpperArm'): [ 255,140,0,1 ],
                 robot.model.getFrameId('RightForeArm'): [ 255,69,0,1 ],
                 robot.model.getFrameId('LeftForeArm'): [ 255,69,0,1 ],
                 robot.model.getFrameId('RightHand'): [ 240,128,128,1 ],
                 robot.model.getFrameId('LeftHand'): [ 240,128,128,1 ] }


# Create marker objects
for m in mnames:
    gv.deleteNode('world/p', True)
    oname = f'world/marker_{m}'  
    gv.addSphere(oname,.01, jointToColor[pairing[m][0]])
    xyz = [markers[m][0][1], -markers[m][0][0], markers[m][0][2]]
    gv.applyConfiguration(oname,xyz+[1,0,0,0])
gv.refresh()

gv.addXYZaxis('world/f1',[1.,1.,1.,1.],.03,.1)

# Ankle Markers - Right & Left Lateral Malleolus
RLM = 1e-3 * np.array(data_markers["RLM"])
LLM = 1e-3 * np.array(data_markers["LLM"])

# Hips Markers - Right & Left Anterior Superior Iliac Spine
RASIS = 1e-3 * np.array(data_markers["RASIS"])
LASIS = 1e-3 * np.array(data_markers["LASIS"])

# Hips Markers - Right & Left Posterior Superior Iliac Spine
RPSIS = 1e-3 * np.array(data_markers["RPSIS"])
LPSIS = 1e-3 * np.array(data_markers["LPSIS"])

# "Waist position", computed as average from markers
waist_exp = (1 / 4) * (RASIS[:] + LASIS[:] + RPSIS[:] + LPSIS[:])


k = 0

q[7:] = np.array( [lumbar_x[k], lumbar_y[k], 0, 0, 0, 0, 0, 0, lumbar_z[k], 
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 
                    left_hip_AA[k], left_hip_FE[k], left_hip_rot[k],
                    left_knee_FE[k], left_knee_rot[k],
                    left_ankle_AA[k], left_ankle_FE[k], left_ankle_rot[k],
                    left_foot_FE[k],
                    right_hip_AA[k], right_hip_FE[k], right_hip_rot[k],
                    right_knee_FE[k], right_knee_rot[k],
                    right_ankle_AA[k], right_ankle_FE[k], right_ankle_rot[k],
                    right_foot_FE[k]])


rot_matrix = pin.rpy.rpyToMatrix(orientation_x[0], orientation_y[0], orientation_z[0])
quat_pelvis = pin.Quaternion(rot_matrix)
q[3:7] = np.array([quat_pelvis[0], quat_pelvis[1], quat_pelvis[2], quat_pelvis[3]])

# q[:3] = np.array([translation_x[k], translation_y[k], translation_z[k]])
q[:3] = np.array([waist_exp[k,1], -waist_exp[k,0], waist_exp[k,2]])

pin.framesForwardKinematics(robot.model,robot.data,q)
pin.updateFramePlacements(robot.model,robot.data)
f = robot.data.oMf[-1]

q[2] -= f.translation[2]

robot.display(q)


while k < len(marche.time):
    time_start = time.time()

    # q[:3] = np.array([translation_x[k], translation_y[k], translation_z[k]])
    q[:3] = np.array([waist_exp[k,1], -waist_exp[k,0], waist_exp[k,2]])

    rot_matrix = pin.rpy.rpyToMatrix(orientation_x[k], orientation_y[k], orientation_z[k])
    quat_pelvis = pin.Quaternion(rot_matrix)
    q[3:7] = np.array([quat_pelvis[0], quat_pelvis[1], quat_pelvis[2], quat_pelvis[3]])

    q[7:] = np.array( [lumbar_x[k], lumbar_y[k], 0, 0, 0, 0, 0, 0, lumbar_z[k], 
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 
                        left_hip_AA[k], left_hip_FE[k], left_hip_rot[k],
                        left_knee_FE[k], left_knee_rot[k],
                        left_ankle_AA[k], left_ankle_FE[k], left_ankle_rot[k],
                        left_foot_FE[k],
                        right_hip_AA[k], right_hip_FE[k], right_hip_rot[k],
                        right_knee_FE[k], right_knee_rot[k],
                        right_ankle_AA[k], right_ankle_FE[k], right_ankle_rot[k],
                        right_foot_FE[k]])

    pin.framesForwardKinematics(robot.model,robot.data,q)
    pin.updateFramePlacements(robot.model,robot.data)
    # f = robot.data.oMf[-1]

    q[2] -= f.translation[2]
    

    for m in mnames:
        xyz = [markers[m][k][1], -markers[m][k][0], markers[m][k][2]]
        oname = f'world/marker_{m}'
        gv.applyConfiguration(oname,xyz+[1,0,0,0])
        gv.refresh()

    if (DISPLAY) and (k % DISPLAY_N == 0):
        robot.display(q)

    time_spent = time.time() - time_start
    if (time_spent < dt) and DISPLAY:
        time.sleep(dt - time_spent + 0.01)
    # time.sleep(0.1)

    if k == 0:
        print(q)
        print(RLM[0, :])
        print(LLM[0, :])
        print(waist_exp[0, :])

    k += 1
    # k = len(marche.time)


########################### PLOT ###########################
if PLOT_JOINTS:
    # Plot example 1: Flexion/extension of the left Hip/Knee/Ankle
    plt.figure(0)

    plt.subplot(221)
    plt.plot(data_kinematics.time, left_hip_AA)
    plt.plot(data_kinematics.time, right_hip_AA)
    plt.xlabel("Time (s)")
    plt.ylabel("Joints positions (rad)")
    plt.title("Hip AA")

    plt.subplot(222)
    plt.plot(data_kinematics.time, left_hip_FE)
    plt.plot(data_kinematics.time, right_hip_FE)
    plt.xlabel("Time (s)")
    plt.ylabel("Joints positions (rad)")
    plt.title("Hip FE")

    plt.subplot(223)
    plt.plot(data_kinematics.time, left_hip_rot)
    plt.plot(data_kinematics.time, right_hip_rot)
    plt.xlabel("Time (s)")
    plt.ylabel("Joints positions (rad)")
    plt.title("Hip Rot")

    plt.subplots_adjust(
        top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
    )
    plt.suptitle("Left leg")

    # Plot example 2: Flexion/extension of the right Hip/Knee/Ankle
    plt.figure(1)

    plt.subplot(221)
    plt.plot(data_kinematics.time, right_hip_FE)
    plt.xlabel("Time (s)")
    plt.ylabel("Joints positions (rad)")
    plt.title("Experimental hip angular position - C.")

    plt.subplot(222)
    plt.plot(data_kinematics.time, right_knee_FE)
    plt.xlabel("Time (s)")
    plt.ylabel("Joints positions (rad)")
    plt.title("Experimental knee angular position - C.")

    plt.subplot(223)
    plt.plot(data_kinematics.time, right_ankle_FE)
    plt.xlabel("Time (s)")
    plt.ylabel("Joints positions (rad)")
    plt.title("Experimental ankle angular position - C.")

    plt.subplots_adjust(
        top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
    )
    plt.suptitle("Right leg")

    plt.show()

    # Plot example 2: Flexion/extension of the right Hip/Knee/Ankle
    plt.figure(2)

    plt.subplot(221)
    plt.plot(data_kinematics.time, radians2degrees(right_hip_AA))
    plt.plot(data_kinematics.time, radians2degrees(left_hip_AA))
    plt.xlabel("Time (s)")
    plt.ylabel("Joints positions (rad)")
    plt.title("Experimental hip angular position - C.")

    plt.subplot(223)
    plt.plot(data_kinematics.time, radians2degrees(right_ankle_AA))
    plt.plot(data_kinematics.time, radians2degrees(right_ankle_AA))
    plt.xlabel("Time (s)")
    plt.ylabel("Joints positions (rad)")
    plt.title("Experimental ankle angular position - C.")

    plt.subplots_adjust(
        top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
    )
    plt.suptitle("Right leg")

    plt.show()

if PLOT_MARKERS:
    # PLot example 3: Feet and Waist positions

    trace_right_foot_exp = go.Scatter3d(
        x=RLM[:, 1], y=-RLM[:, 0], z=RLM[:, 2], mode="lines", name="Right Foot"
    )

    trace_left_foot_exp = go.Scatter3d(
        x=LLM[:, 1], y=-LLM[:, 0], z=LLM[:, 2], mode="lines", name="Left Foot"
    )

    trace_com_exp = go.Scatter3d(
        x=waist_exp[:, 1],
        y=-waist_exp[:, 0],
        z=waist_exp[:, 2],
        mode="lines",
        name="Com",
    )

    trace_pelvis_exp = go.Scatter3d(
        x=np.array(translation_x[:]),
        y=np.array(translation_y[:]),
        z=np.array(translation_z[:]),
        mode="lines",
        name="Pelvis",
    )

    x_start = [RLM[0, 1], LLM[0, 1], waist_exp[0, 1]]
    y_start = [-RLM[0, 0], -LLM[0, 0], -waist_exp[0, 0]]
    z_start = [RLM[0, 2], LLM[0, 2], waist_exp[0, 2]]

    x_hs = []
    y_hs = []
    z_hs = []
    for iter in t_hs_right:
        nb = int(iter / dt)
        x_hs.append(RLM[nb, 1])
        y_hs.append(-RLM[nb, 0])
        z_hs.append(RLM[nb, 2])
    for iter in t_hs_left:
        nb = int(iter / dt)
        print(nb, t_hs_left)
        x_hs.append(LLM[nb, 1])
        y_hs.append(-LLM[nb, 0])
        z_hs.append(LLM[nb, 2])

    x_to = []
    y_to = []
    z_to = []
    for iter in t_to_right:
        nb = int(iter / dt)
        x_to.append(RLM[nb, 1])
        y_to.append(-RLM[nb, 0])
        z_to.append(RLM[nb, 2])
    for iter in t_to_left:
        nb = int(iter / dt)
        x_to.append(LLM[nb, 1])
        y_to.append(-LLM[nb, 0])
        z_to.append(LLM[nb, 2])

    trace_start = go.Scatter3d(
        x=x_start, y=y_start, z=z_start, mode="markers", name="Start"
    )

    trace_heel_strike = go.Scatter3d(
        x=x_hs, y=y_hs, z=z_hs, mode="markers", name="Heel Strike"
    )

    trace_toes_off = go.Scatter3d(
        x=x_to, y=y_to, z=z_to, mode="markers", name="Toes Off"
    )

    name = "eye = (x:0., y:2.5, z:0.)"
    camera = dict(
        up=dict(x=0, y=0.0, z=1.0),
        eye=dict(x=0, y=-3, z=0.3),
        center=dict(x=0, y=0.0, z=-0.5),
    )

    fig = go.Figure(
        data=[
            trace_right_foot_exp,
            trace_left_foot_exp,
            trace_com_exp,
            trace_pelvis_exp,
            trace_start,
            trace_heel_strike,
            trace_toes_off,
        ]
    )
    fig.update_layout(
        scene_aspectmode="data",
        scene_camera=camera,
        legend=dict(font=dict(size=20), yanchor="top", y=0.8, xanchor="right", x=0.9),
    )  # , xaxis = dict(tickfont = dict(size=20)), yaxis = dict(tickfont = dict(size=20)))
    fig.update_yaxes(tickfont=dict(size=50))
    fig.update_traces(marker=dict(size=3))
    fig.show()
