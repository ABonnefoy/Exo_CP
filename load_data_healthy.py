##
## Copyright (c) 2022, LAAS-CNRS ???
##
## Authors: Aurélie Bonnefoy, Sabrina Otmani
##          refactor Nicolas Mansard, based on data_healthy_visu.
##
## This file is part of an ongoing work on the interaction of an exoskeleton and a human suffering prom cerebral palsy
##
## Define the class MocapData that loads the model and post-process the mocap data.
## The class containts:
##    - model: a pinocchio model (also contains a full robot object)
##    - configurations: the list of recorded (mocap) configurations for the model
##    - markers: raw marker datas
##    - contact_patterns: array (same length as configurations) with booleans denoting contacts
##    - reduceModel() to cast the model to a simpler one (frozen upper body)
##    - some visualization tricks
##
## See the main at the end of the file for an example of how to use it.

import numpy as np
import time
import pinocchio as pin
import os
from mat4py import loadmat
import math
from pinocchio.visualize import GepettoVisualizer
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

def save_visu_data(path, waist_se3, right_foot_se3, left_foot_se3):
    np.savez(
        path + "/new_data/" + studied_walk + "_refs_from_visu.npz",
        waist_se3=waist_se3,
        right_foot_se3=right_foot_se3,
        left_foot_se3=left_foot_se3
    )


####### End of Functions creation ######

class MocapData:
    '''
    The class containts:
       - model: a pinocchio model (also contains a full robot object)
       - configurations: the list of recorded (mocap) configurations for the model
       - markers: raw marker datas
       - contact_patterns: array (same length as configurations) with booleans denoting contacts
       - reduceModel() to cast the model to a simpler one (frozen upper body)
       - some visualization tricks
    '''
    def __init__(self):
        # PATH & URD
        #path = os.path.dirname(os.path.realpath(__file__))
        path = '.'
        urdf = path + "/urdf/healthy_human.urdf"
        
        # Creation du robot pinocchio
        self.robot = robot = pin.RobotWrapper.BuildFromURDF(
            urdf, package_dirs=None, root_joint=pin.JointModelFreeFlyer(), verbose=False
        )


        # Configuration
        self.model = model = robot.model
        self.data = data = model.createData()
        q = np.zeros(model.nq)

        t = 0.0  # time
        self.dt = dt = 0.005

        ########################### READ DATA (kinematics, moments, events) #############

        studied_walk = "marche 1"
        studied_markers = "markers_data"
        delay = 255 * dt
        
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
        self.marche = marche = marches.Marche(
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
        self.t_hs_left = t_hs_left = np.array(marche.heelstrike_left) - delay
        self.t_hs_right = t_hs_right = np.array(marche.heelstrike_right) - delay
        self.t_to_left = t_to_left = np.array(marche.toesoff_left) - delay
        self.t_to_right = t_to_right = np.array(marche.toesoff_right) - delay
        # # Add dummy value at the end of the list 
        # t_hs_left = np.append(t_hs_left, np.zeros(1))
        # t_hs_right = np.append(t_hs_right, np.zeros(1))
        # t_to_left = np.append(t_to_left, np.zeros(1))
        # t_to_right = np.append(t_to_right, np.zeros(1))

        i_hs_left = 0
        i_hs_right = 0
        i_to_left = 0
        i_to_right = 0
        
        # Read markers
        data_markers_global = loadmat(path + "/walk_data/" + studied_markers + ".mat")
        data_markers = data_markers_global["output"]["marker_data"]["Markers"]
        
        self.markers = markers = { m:  1e-3*np.array(l) for m,l in data_markers.items() }
        mnames = list(markers.keys())
        self.T = T = len(markers[mnames[0]])
        # Choose initial timing as second heel strike
        self.T0 = int(np.ceil(max(self.t_hs_right[0],self.t_hs_left[0])/dt))
        
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

        ### POST TREAT CONFIG DATA
        k = 0
        delta_z = 0
        configurations = []
        contact_patterns = []
        times = []
        
        nphase_right = 0
        nphase_left = 0
        
        self.contactIds = [ 'RightFoot', 'LeftFoot' ]
        contacts = [ False, False ] # Both feet assumed not in contact at start

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
            
            if k == 0:
                pin.framesForwardKinematics(robot.model,robot.data,q)
                pin.updateFramePlacements(robot.model,robot.data)
                delta_z = robot.data.oMf[-1].translation[2]

            q[2] -= delta_z

            # Evaluating contact phases, based on HS (heel strike) and TO (toe off)
            # timings
            if i_hs_right < len(t_hs_right) and t < t_hs_right[i_hs_right] <= t + dt:
                i_hs_right+=1
                contacts[0] = True
                
            if i_to_right < len(t_to_right) and t < t_to_right[i_to_right] <= t + dt:
                i_to_right+=1
                contacts[0] = False
                
            if i_hs_left < len(t_hs_left) and t < t_hs_left[i_hs_left] <= t + dt:
                i_hs_left+=1
                contacts[1] = True
        
            if i_to_left < len(t_to_left) and t < t_to_left[i_to_left] <= t + dt:
                i_to_left+=1
                contacts[1] = False

            k += 1
            # k = len(marche.time)
            t+=dt

            times.append(t)
            configurations.append(q.copy())
            contact_patterns.append(contacts.copy())

        self.times = np.array(times)
        self.configurations = np.array(configurations)
        self.contact_patterns = np.array(contact_patterns)

    def markersDisplayInfo(self):
        robot = self.robot
        ## Pairing of markers
        self.pairing = {
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

        self.jointToColor = { robot.model.getFrameId('Pelvis'): [ 50,205,50,1 ],
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


    def computeOperationTrajectories(self):
        '''
        Compute the basis and feet trajectories from configuration trajectory.
        Store the result in waist_se3, {right|left}_foot_se3.
        '''
        self.waist_se3= []
        self.right_foot_se3= []
        self.left_foot_se3= []
        data = self.model.createData()
        for q in self.configurations:
            pin.framesForwardKinematics(self.model,data,q)
            self.waist_se3.append(data.oMf[self.model.getFrameId("Pelvis")].copy())
            self.right_foot_se3.append(data.oMf[self.model.getFrameId("RightFoot")].copy())
            self.left_foot_se3.append(data.oMf[self.model.getFrameId("LeftFoot")].copy())
            
    def initDisplay(self):
        self.markersDisplayInfo()
        self.robot.initViewer(loadModel=True)
        gv=self.gv=self.robot.viewer.gui
        self.gv.addFloor("world/floor")

        # Create marker objects
        for m in self.markers.keys():
            oname = f'world/marker_{m}'  
            gv.addSphere(oname,.01, self.jointToColor[self.pairing[m][0]])
            xyz = [self.markers[m][0][1], -self.markers[m][0][0], self.markers[m][0][2]]
            gv.applyConfiguration(oname,xyz+[1,0,0,0])
            gv.refresh()

        gv.addXYZaxis('world/f1',[1.,1.,1.,1.],.03,.1)
        gv.addBox('world/right_contact',.18,.1,0.03, [1, 0, 0, 1])
        gv.addBox('world/left_contact',.18,.1,0.03, [0, 1, 0, 1])
        self.display(self.T0)

    def reduceModel(self):
        '''
        Remove upper torso, head and arms
        '''
        # Joint id to freeze
        toFreeze = list(range(4,32))+[40,49]
        # Indexes of the configuration vector to keep
        self.iqs = iqs = []
        for i,(iq,nq) in enumerate(zip(self.model.idx_qs,self.model.nqs)):
            if i not in toFreeze:
                iqs.extend(range(iq,iq+nq))
        # Save old models, just in case
        self.modelXL = self.model
        self.robot.collision_modelXL = self.robot.collision_model
        self.robot.visual_modelXL = self.robot.visual_model
        # Reduce model
        self.model, [self.robot.collision_model,self.robot.visual_model] = \
            pin.buildReducedModel(self.model,[self.robot.collision_model,self.robot.visual_model],
                                  toFreeze,self.robot.q0)
        self.robot.model = self.model
        self.robot.rebuildData()
        if self.robot.viz is not None:
            self.robot.viz.rebuildData()
        # Resize configuration array
        self.robot.q0 = self.robot.q0[iqs]
        self.configurationsXL = self.configurations
        self.configurations = self.configurations[:,iqs]

        
    def setTransparentDisplay(self,alpha=.6):
        for g in self.robot.visual_model.geometryObjects:
            n=self.robot.viz.getViewerNodeName(g,pin.VISUAL)
            self.gv.setFloatProperty(n,'Alpha',alpha)
            
    def display(self,k,sleep='auto'):
        gv=self.gv; robot=self.robot
        for m in self.markers.keys():
            xyz = [self.markers[m][k][1], -self.markers[m][k][0], self.markers[m][k][2]]
            oname = f'world/marker_{m}'
            gv.applyConfiguration(oname,xyz+[1,0,0,0])

        q = self.configurations[k]

        # Display foot contacts
        pin.updateFramePlacements(self.model,robot.viz.data)
        # Right contact
        gv.setVisibility('world/right_contact', 'ON' if self.contact_patterns[k][0] else 'OFF')
        rf = pin.SE3ToXYZQUAT(robot.viz.data.oMf[robot.model.getFrameId("RightFoot")]).tolist()
        rf[3:] = [1,0,0,0]
        gv.applyConfiguration('world/right_contact',rf)
        # Left contact
        gv.setVisibility('world/left_contact', 'ON' if self.contact_patterns[k][1] else 'OFF')
        lf = pin.SE3ToXYZQUAT(robot.viz.data.oMf[robot.model.getFrameId("LeftFoot")]).tolist()
        lf[3:] = [1,0,0,0]
        gv.applyConfiguration('world/left_contact',lf)

        robot.display(q)
        
        if sleep=='auto': sleep = self.dt
        if sleep>0: time.sleep(sleep)

    def diff(self):
        '''
        Compute the numerical derivatives of the configuration vq and aq
        '''
        import scipy.ndimage
        dqs = [ scipy.ndimage.gaussian_filter1d(qi,3,order=1) for qi in self.configurations.T ]
        vs = dqs[:3]
        ws = [ np.zeros(self.configurations.shape[0]) ]*3
        vqs = dqs[7:]
        self.vqs = np.array(vs+ws+vqs).T/self.dt

        dqs = [ scipy.ndimage.gaussian_filter1d(qi,2.5,order=2) for qi in self.configurations.T ]
        vs = dqs[:3]
        ws = [ np.zeros(self.configurations.shape[0]) ]*3
        vqs = dqs[7:]
        self.aqs = np.array(vs+ws+vqs).T/self.dt

        
        #vqs = np.array([ scipy.ndimage.gaussian_filter1d(qi[7:],3,order=1) for qi in self.configurations.T ]).T/self.dt
        #self.vqs = np.hstack([vs,ws,vqs])
        #self.aqs = np.array([ scipy.ndimage.gaussian_filter1d(qi,2.5,order=2) for qi in self.configurations.T ]).T/self.dt**2

        
if __name__ == "__main__":
    data = MocapData()
    data.reduceModel()
    data.initDisplay()
    data.setTransparentDisplay()
    for k in range(data.T0,data.T):
        data.display(k)

