import pinocchio as pin
import time

def viewerTarget(model,collision_model,visual_model,alpha=.4):
    '''
    Create a second viewer for the model in transparent style.
    '''
    viz = pin.visualize.GepettoVisualizer(model,collision_model,visual_model)
    viz.initViewer(loadModel=True,sceneName='target')
    for g in visual_model.geometryObjects:
        name=viz.getViewerNodeName(g,pin.VISUAL)
        viz.viewer.gui.setFloatProperty(name,'Alpha',alpha)
    return viz

class DoublePlayer:
    def __init__(self,v1,v2):
        self.v1 = v1
        self.v2 = v2
    def play(self,qs1,qs2,dt):
        for q1,q2 in zip(qs1.T,qs2.T): ## .T to match the API of viz.play
            self.v1.display(q1)
            self.v2.display(q2)
            time.sleep(dt)
    def __call__(self,*a):
        self.play(*a)
