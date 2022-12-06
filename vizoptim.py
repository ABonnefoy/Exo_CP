'''
OCP with constrained dynamics

Simple example with regularization cost and terminal com+velocity constraint, known initial config.

min X,U,A,F    sum_t  
                (q-q0)**2 + v**2 ## state regularisation
                + vcom**2
                + sum_foot cop**2
                        + distance(f,cone central axis)
                        + (f-f_t^*)**2

s.t
        x_0 = x0
        x_t+1  = EULER(x_t,u_t,f_t |  f=a_t )
        a_t = aba(q_t,v_t,u_t)
        a_feet(t) = 0
        v_T = x_T [nq:]  = 0
        com(q_t)[2] = com(x_T[:nq])[2] = X_TARGET

        forall t:
                forall foot: 
                        v_foot**2/v_0 < z_foot/z_0      ### Fly high
                        cop \in foot_box                ### COP constraint
                        f_z >= 0                        ### Not pull on ground
                        z_foot>=0                       ### not ground collision
                distance(right foot, left foot) >= 17cm
        forall impact ti:
                z_foot = 0
                v_foot = 0_6
                roll_foot = pitch_foot = 0

So the robot should just bend to reach altitude COM 80cm while stoping at the end of the movement.
The acceleration is introduced as an explicit (slack) variable to simplify the formation of the contact constraint.
Contacts are 6d, yet only with normal forces positive, and 6d acceleration constraints. 


'''

import pinocchio as pin
import casadi
import numpy as np
import example_robot_data as robex
import matplotlib.pyplot as plt; #plt.ion()
from numpy.linalg import norm,inv,pinv,svd,eig
# Local imports
from weight_share import computeReferenceForces

pin.SE3.__repr__=pin.SE3.__str__
np.set_printoptions(precision=2, linewidth=300, suppress=True,threshold=10000)

from load_data_healthy import MocapData
import vtarget

mocap = MocapData()
mocap.reduceModel()
mocap.diff()
model = mocap.model
mocap.T = int(np.floor(max(mocap.t_to_right[1],mocap.t_to_left[1])/mocap.dt))


print('For display purposes, please run gepetto-gui before running this code')

try:
    viz = pin.visualize.GepettoVisualizer(model,mocap.robot.collision_model,mocap.robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
    viz2 = vtarget.viewerTarget(model,mocap.robot.collision_model,mocap.robot.visual_model,.1)
    play = vtarget.DoublePlayer(viz,viz2)
except:
    print("No viewer"  )



# The pinocchio model is what we are really interested by.
model = model
model.q0 = mocap.configurations[0] # robot.q0

GUESS_FILE = './sol.npy'
# GUESS_FILE = './sol_patho.npy'
guess = np.load(GUESS_FILE,allow_pickle=True)[()]


xs = guess['xs']
us = guess['us']
fs = guess['fs']

qs = xs[:,:model.nq]
print(model.nq)
vqs = xs[:,model.nq:]
qs_ref = mocap.configurations[mocap.T0:mocap.T]

play(qs.T,qs_ref.T,mocap.dt)


font = {'size': 20}
plt.rc('font', **font)

plt.figure(0)

for i in range(6,us.shape[1]):
        plt.plot(us[:,i], label = str(i))
plt.legend()
plt.xlabel("iteration (dt = 5ms)")
plt.ylabel("Torque [N.m]")
plt.title("Human Torques - H")

plt.show()
