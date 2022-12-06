##
## Copyright (c) 2022, LAAS-CNRS ???
##
## Authors: Aurélie Bonnefoy, Sabrina Otmani
##
## This file is part of an ongoing work on the interaction of an exoskeleton and a human suffering from cerebral palsy
## 
## Use: Check that the kinematics of the exoskeleton is working with the human
##
##
##


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

from scipy.optimize import fmin_bfgs, fmin_slsqp, minimize
import quadprog

# Used for debug
from IPython import embed
import sys


def compute_G(model, modelx, qh, qx):

    Sh = np.eye(model.nv)
    Sh = Sh[:,6:]

    ax_joints = [6,9,10,12,15,16]
    Sx = np.zeros((len(ax_joints), modelx.nv))
    for i in range(len(ax_joints)):
        Sx[i,ax_joints[i]] = 1

    Sx = Sx.T

    data = model.createData()
    Jh_waist = pin.computeFrameJacobian(model, data, qh, model.getFrameId('Pelvis'), pin.LOCAL_WORLD_ALIGNED)
    Jh_rf = pin.computeFrameJacobian(model, data, qh, model.getFrameId('RightFoot'), pin.LOCAL_WORLD_ALIGNED)
    Jh_lf = pin.computeFrameJacobian(model, data, qh, model.getFrameId('LeftFoot'), pin.LOCAL_WORLD_ALIGNED)
    Jh = np.vstack((Jh_waist, Jh_rf, Jh_lf))

    datax = modelx.createData()
    Jx_waist = pin.computeFrameJacobian(modelx, datax, qx, modelx.getFrameId('millieu_hanche'), pin.LOCAL_WORLD_ALIGNED)
    Jx_rf = pin.computeFrameJacobian(modelx, datax, qx, modelx.getFrameId('pied_2'), pin.LOCAL_WORLD_ALIGNED)
    Jx_lf = pin.computeFrameJacobian(modelx, datax, qx, modelx.getFrameId('pied'), pin.LOCAL_WORLD_ALIGNED)
    Jx = np.vstack((Jx_waist, Jx_rf, Jx_lf))
    assert(Jh.shape[0]==Jx.shape[0])
    assert(Jh.T.shape[1]==3*6)

    lG = Sx.shape[1] + Sh.shape[1]
    cG = Sh.shape[0] + Jh.shape[0] + Sx.shape[0]

    G = np.zeros((lG,cG))


    G[:Sh.shape[1],:Sh.shape[0]] = Sh.T.copy()
    G[:Sh.shape[1],Sh.shape[0]:Sh.shape[0]+Jh.shape[0]] = Jh[:,6:].T.copy()

    G[Sh.shape[1]:,Sh.shape[0]:Sh.shape[0]+Jx.shape[0]] = Jx[:,ax_joints].T.copy()
    G[Sh.shape[1]:,Sh.shape[0]+Jx.shape[0]:] = Sx.T.copy()
    
    return(G)


    

    

def compute_g(model, modelx, qh, dqh, ddqh, qx, dqx, ddqx, fc):

    data = model.createData()
    pin.rnea(model, data, qh, dqh, ddqh)
    rnea_h = data.tau
    

    Jcr = pin.computeFrameJacobian(model, data, qh, model.getFrameId('RightFoot'), pin.LOCAL)
    Jcl = pin.computeFrameJacobian(model, data, qh, model.getFrameId('LeftFoot'), pin.LOCAL)
    assert(Jcr.shape==Jcl.shape)
    Jc = np.vstack((Jcr, Jcl))

    tauc = Jc.T @ fc

    assert(rnea_h.shape[0]==tauc.shape[0])

    datax = modelx.createData()
    pin.rnea(modelx, datax, qx, dqx, ddqx)
    rnea_x = datax.tau
    print(rnea_x)

    pin.computeJointJacobians(model, data, qh)
    pin.computeJointJacobians(modelx, datax, qx)
    Jh = data.J.copy()
    Jx = datax.J.copy()
    assert(Jh.shape[0]==Jx.shape[0])

    ax_joints = [6,9,10,12,15,16]
    lg = rnea_h.shape[0]-6+len(ax_joints)
    g = np.zeros(lg)

    g[:rnea_h.shape[0]-6] = rnea_h[6:] - tauc[6:]
    g[rnea_h.shape[0]-6:] = rnea_x[ax_joints].copy()

    return(g)


def cost(x):
    x[3:7] /= norm(x[3:7])


    data = model.createData()    
    pin.framesForwardKinematics(model,data,q_ref)
    pin.updateFramePlacements(model,data)

    pos_target_right_foot = data.oMf[model.getFrameId('RightFoot')]
    pos_target_left_foot=data.oMf[model.getFrameId('LeftFoot')]
    pos_target_pelvis=data.oMf[model.getFrameId('Pelvis')]

    datax = modelx.createData()
    pin.framesForwardKinematics(modelx,datax,x)
    pin.updateFramePlacements(modelx,datax)

    pos_exo_right_foot = datax.oMf[modelx.getFrameId('pied_2')]
    pos_exo_left_foot=datax.oMf[modelx.getFrameId('pied')]
    pos_exo_pelvis=datax.oMf[modelx.getFrameId('millieu_hanche')]

    Rz_foot=pin.SE3(pin.utils.rotate('z',np.pi/2),np.zeros(3))*pin.SE3(pin.utils.rotate('y',-np.pi/2),np.zeros(3))
    return np.sum(pin.log((pos_exo_right_foot*Rz_foot).inverse()*pos_target_right_foot).vector**2) + np.sum(pin.log((pos_exo_left_foot*Rz_foot).inverse()*pos_target_left_foot).vector**2) + np.sum(pin.log((pos_exo_pelvis).inverse()*pos_target_pelvis).vector**2) #+ np.sum(pin.log((pos_exo_right_thigh).inverse()*pos_target_right_thigh).vector**2)+ np.sum(pin.log((pos_exo_left_thigh).inverse()*pos_target_left_thigh).vector**2))


def quadprog_solve_qp(P, q, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            A * x == b

    using quadprog <https://pypi.python.org/pypi/quadprog/>.

    Parameters
    ----------
    P : numpy.array
        Symmetric quadratic-cost matrix.
    q : numpy.array
        Quadratic-cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : numpy.array, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector (not used).

    Returns
    -------
    x : numpy.array
        Solution to the QP, if found, otherwise ``None``.
    """

    if initvals is not None:
        print("quadprog: note that warm-start values ignored by wrapper")
    qp_G = P
    qp_a = -q
    qp_C = -A.T
    qp_b = -b

    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b)[0]

def diff(configurations):
    '''
    Compute the numerical derivatives of the configuration vq and aq
    '''
    import scipy.ndimage
    dqs = [ scipy.ndimage.gaussian_filter1d(qi,3,order=1) for qi in configurations.T ]
    vs = dqs[:3]
    ws = [ np.zeros(configurations.shape[0]) ]*3
    vqs = dqs[7:]
    vqs_ = np.array(vs+ws+vqs).T/dt

    dqs = [ scipy.ndimage.gaussian_filter1d(qi,2.5,order=2) for qi in configurations.T ]
    vs = dqs[:3]
    ws = [ np.zeros(configurations.shape[0]) ]*3
    vqs = dqs[7:]
    aqs = np.array(vs+ws+vqs).T/dt

    return(vqs_, aqs)



if __name__ == "__main__":

    dt = 0.005
    
    mocap = MocapData()
    mocap.reduceModel()
    mocap.diff()
    model = mocap.model
    mocap.T = int(np.floor(max(mocap.t_to_right[1],mocap.t_to_left[1])/mocap.dt))

    print('For display purposes, please run gepetto-gui before running this code')


    # The pinocchio model is what we are really interested by.
    robot = mocap.robot
    model = model
    model.q0 = mocap.configurations[0] # robot.q0
    robot.initViewer(loadModel=True)

    GUESS_FILE = './sol.npy'
    # GUESS_FILE = './sol_patho.npy'
    guess = np.load(GUESS_FILE,allow_pickle=True)[()]


    xs = guess['xs']
    us = guess['us']
    fs = guess['fs']
    aqs = guess['acs']
    print(np.amax(aqs))
    qs = xs[:,:model.nq]
    vqs = xs[:,model.nq:]
    print(np.amax(vqs))
    qs_ref = mocap.configurations[mocap.T0:mocap.T]

    ### Load exoskeleton

    path = '.'

    urdf = path + "/urdf/healthy_human.urdf"
    urdf_exo = path + "/urdf/humanoid_complet/robot.urdf"


    exo = pin.RobotWrapper.BuildFromURDF(
        urdf_exo, package_dirs=path + "/urdf/humanoid_complet/", root_joint=pin.JointModelFreeFlyer(), verbose=False)
    modelx = exo.model
    datax = exo.data

    exo.initViewer(loadModel=True)
    exo.display(exo.q0)
    q_exo = exo.q0.copy()

    ## Optimization phase

    q_exo_list = []
    v_exo_list = []
    a_exo_list = []
    
    # sys.exit()
    for i in range(len(qs)-1):

        q_ref=qs[i,:]
        robot.display(q_ref)
        x0 = q_exo.copy()
        
        ## Angular bounds for each joint

        a_bounds=[(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000),(-1000,1000)]
        a_bounds[8]=(-1,1)      #Right hip rotation
        a_bounds[9]=(-0.2,0.2)  #Right hip adduction
        a_bounds[10]=(0,100)
        a_bounds[12]=(-1,1)     #Right Ankle rotation
        a_bounds[14]=(-0.3,0.3) #Left Hip rotation
        a_bounds[15]=(-0.3,0.3) #Left Hip adduction
        a_bounds[18]=(-0.3,0.3) #Left Ankle rotation 
        a_bounds[16]=(0,1000)

        ## Optimization phase

        # xopt = fmin_bfgs(cost, x0)
        xopt = fmin_slsqp(cost, x0, bounds=a_bounds) #, iprint=2, full_output=1)[0]

        assert xopt.shape == (exo.nq,)
        xopt[3:7]=xopt[3:7]/norm(xopt[3:7]) ## Normalisation

        ## Update of the position/orientation of the exoskeleton

        q_exo=xopt.copy()
        exo.display(xopt)

        q_exo_list.append(q_exo)

    q_exo_array = np.array(q_exo_list)
    v_exo_array, a_exo_array = diff(q_exo_array)

    tau_h = []
    f_i = []
    tau_x = [] 
    fail = [] 


    for k in range(len(qs)-1):

        b = compute_g(model, modelx, qs[k], vqs[k], aqs[k], q_exo_array[k], v_exo_array[k], a_exo_array[k], fs[k])    
        A = compute_G(model, modelx, qs[k], q_exo_array[k])
        

        P = np.eye(robot.nv+3*6+exo.nv)
        try:
            min_vector = quadprog_solve_qp(P=P, q=np.zeros(robot.nv+3*6+exo.nv), A=A,b=b)
            tau_h.append(min_vector[:robot.nv].copy())
            f_i.append(min_vector[robot.nv:robot.nv+3*6].copy())
            tau_x.append(min_vector[robot.nv+3*6:].copy())

        except:
            print('non-working')
            fail.append(k)
            tau_h.append([])
            f_i.append([])
            tau_x.append([])
        print(k)

    tau_h_array = np.array(tau_h)
    f_i_array = np.array(f_i)
    tau_x_array = np.array(tau_x)


    ### PLOTTING ###


    font = {'size': 20}
    plt.rc('font', **font)

    
    plt.figure(0)

    for i in range(6,tau_h_array.shape[1]):
        plt.plot(tau_h_array[:,i], label = str(i))
    plt.legend()
    plt.xlabel("iteration (dt = 5ms)")
    plt.ylabel("Torque [N.m]")
    plt.title("Human Torques")
    plt.rc('font', **font)

    plt.show()

    plt.figure(1)

    for i in range(6,tau_x_array.shape[1]):
        plt.plot(tau_x_array[:,i], label = str(i))
    plt.legend()
    plt.xlabel("iteration (dt = 5ms)")
    plt.ylabel("Torque [N.m]")
    plt.title("Exoskeleton Torques")
    plt.rc('font', **font)

    plt.show()

    plt.figure(2)

    plt.plot(norm(f_i_array[:,:3], axis=1), label = 'Waist')
    plt.plot(norm(f_i_array[:,6:9], axis=1), label = 'RightFoot')
    plt.plot(norm(f_i_array[:,12:], axis=1), label = 'LeftFoot')
    plt.legend()
    plt.xlabel("iteration (dt = 5ms)")
    plt.ylabel("Force [N]")
    plt.title("Interaction forces")
    plt.rc('font', **font)

    plt.show()

    plt.figure(3)

    plt.plot(norm(f_i_array[:,3:6], axis=1,), label = 'Waist')
    plt.plot(norm(f_i_array[:,9:12], axis=1), label = 'RightFoot')
    plt.plot(norm(f_i_array[:,15:], axis=1), label = 'LeftFoot')
    plt.legend()
    plt.xlabel("iteration (dt = 5ms)")
    plt.ylabel("Torque [N.m]")
    plt.title("Interaction torques")
    plt.rc('font', **font)

    plt.show()


