##
## Copyright (c) 2022, LAAS-CNRS ???
##
## Authors: Aurélie Bonnefoy, Sabrina Otmani
##
## This file is part of an ongoing work on the interaction of an exoskeleton and a human suffering prom cerebral palsy
## Here, the goal is to use TSID to maintain the starting posture of a pathological gait
##
##
##
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
from sklearn.metrics import r2_score

from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer

# Used for debug
from IPython import embed
import sys



###### Functions Creation ######


def degrees2radians(degrees):
    return degrees * math.pi / 180


def radians2degrees(radians):
    return radians * 180 / math.pi


def torsion(S, D, dt, pos_ref, pos_real):
    torque_torsion = S * (pos_ref - pos_real) + D * (pos_ref - pos_real) / dt
    return torque_torsion


####### End of Functions creation ######




t = 0.0  # time
dt = 0.005
i = 0

########################### READ REFS) ###########################

# Read refs
studied_walk = "marche 1"
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

refs = np.load(path + "/new_data/" + studied_walk + "_refs.npz", allow_pickle=True)

q_list = np.array(refs["q"], np.double)

time_list = refs["time"]

t_hs_left = refs["t_hs_left"]
t_hs_right = refs["t_hs_right"]
t_to_left = refs["t_to_left"]
t_to_right = refs["t_to_right"]

i_hs_left = 0
i_hs_right = 0
i_to_left = 0
i_to_right = 0


k=0
# while t < t_hs_right[0]:
#     k += 1
#     t += dt

# i_hs_right+=1
# i_hs_left+=1

markers = refs["markers"].item()


left_foot_marker = (1/2) * (markers["LSPH"]+markers["LLM"]) 
right_foot_marker = (1/2) * (markers["RSPH"]+markers["RLM"])

assert(left_foot_marker.shape[0] == 3)
assert(right_foot_marker.shape[0] == 3)

left_hand_marker = (1/2) * (markers["LRS"]+markers["LUS"])
right_hand_marker = (1/2) * (markers["RRS"]+markers["RUS"])

assert(left_hand_marker.shape[0] == 3)
assert(right_hand_marker.shape[0] == 3)

waist_marker = (1 / 4) * (markers["RASIS"] + markers["LASIS"] +
                          markers["RPSIS"] + markers["LPSIS"])

assert(waist_marker.shape[0] == 3)

# i = 218

i = len(time_list)
########################### PLOT ###########################

trace_right_foot_exp = go.Scatter3d(
    x=right_foot_marker[0,k:i-2],
    y=right_foot_marker[1,k:i-2],
    z=right_foot_marker[2,k:i-2],
    mode="lines",
    name="Right Foot - experimental",
)

trace_left_foot_exp = go.Scatter3d(
    x=left_foot_marker[0,k:i-2],
    y=left_foot_marker[1,k:i-2],
    z=left_foot_marker[2,k:i-2],
    mode="lines",
    name="Left Foot - experimental",
)

trace_com_exp = go.Scatter3d(
    x=waist_marker[0, k : i - 2],
    y=waist_marker[1, k : i - 2],
    z=waist_marker[2, k : i - 2],
    mode="lines",
    name="Pelvis - experimental",
)

x_start = [right_foot_marker[0,k], left_foot_marker[0,k], waist_marker[0, k]]
y_start = [right_foot_marker[1,k], left_foot_marker[1,k], waist_marker[1, k]]
z_start = [right_foot_marker[2,k], left_foot_marker[2,k], waist_marker[2, k]]

x_hs = []
y_hs = []
z_hs = []

x_to = []
y_to = []
z_to = []

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
        trace_start,
        trace_heel_strike,
        trace_toes_off,
    ]
)
fig.update_layout(
    scene_aspectmode="manual",
    scene_aspectratio=dict(x=2, y=1, z=1),
    scene_camera=camera,
    legend=dict(font=dict(size=20), yanchor="top", y=0.8, xanchor="right", x=0.9),
)  # , xaxis = dict(tickfont = dict(size=20)), yaxis = dict(tickfont = dict(size=20)))
fig.update_yaxes(tickfont=dict(size=50))
fig.update_traces(marker=dict(size=3))
fig.show()



# if PLOT_JOINT_POS:

#     sim_size = len(left_hip_computed[:i])

#     # Côté gauche !
#     plt.figure(0)

#     plt.subplot(221)
#     plt.plot(time_list[:i], radians2degrees(left_hip_computed[:i]))
#     plt.plot(time_list[:i], radians2degrees(q_list[7 + 31, 0]) * np.ones(sim_size))
#     plt.xlabel("Time (s)")
#     plt.ylabel("Angles (degrees)")
#     plt.title("Computed and experimental hip angular position - C.")
#     plt.legend(["Computed", "Experimental"])

#     plt.subplot(222)
#     plt.plot(time_list[:i], radians2degrees(left_knee_computed[:i]))
#     plt.plot(time_list[:i], radians2degrees(q_list[7 + 33, 0]) * np.ones(sim_size))
#     plt.xlabel("Time (s)")
#     plt.ylabel("Angles (degrees)")
#     plt.title("Computed and experimental knee angular position - C.")
#     plt.legend(["Computed", "Experimental"])

#     plt.subplot(223)
#     plt.plot(time_list[:i], radians2degrees(left_ankle_computed[:i]))
#     plt.plot(time_list[:i], radians2degrees(q_list[7 + 36, 0]) * np.ones(sim_size))
#     plt.xlabel("Time (s)")
#     plt.ylabel("Angles (degrees)")
#     plt.title("Computed and experimental ankle angular position - C.")
#     plt.legend(["Computed", "Experimental"])

#     plt.subplots_adjust(
#         top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
#     )
#     plt.suptitle("Left leg")

#     # Côté droit !
#     plt.figure(1)

#     plt.subplot(221)
#     plt.plot(time_list[:i], radians2degrees(right_hip_computed[:i]))
#     plt.plot(time_list[:i], radians2degrees(q_list[7 + 40, 0]) * np.ones(sim_size))
#     plt.xlabel("Time (s)")
#     plt.ylabel("Angles (degrees)")
#     plt.title("Computed and experimental hip angular position - C.")
#     plt.legend(["Computed", "Experimental"])

#     plt.subplot(222)
#     plt.plot(time_list[:i], radians2degrees(right_knee_computed[:i]))
#     plt.plot(time_list[:i], radians2degrees(q_list[7 + 42, 0]) * np.ones(sim_size))
#     plt.xlabel("Time (s)")
#     plt.ylabel("Angles (degrees)")
#     plt.title("Computed and experimental knee angular position - C.")
#     plt.legend(["Computed", "Experimental"])

#     plt.subplot(223)
#     plt.plot(time_list[:i], radians2degrees(right_ankle_computed[:i]))
#     plt.plot(time_list[:i], radians2degrees(q_list[7 + 45, 0]) * np.ones(sim_size))
#     plt.xlabel("Time (s)")
#     plt.ylabel("Angles (degrees)")
#     plt.title("Computed and experimental ankle angular position - C.")
#     plt.legend(["Computed", "Experimental"])

#     plt.subplots_adjust(
#         top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
#     )

#     plt.suptitle("Right leg")

#     plt.show()
