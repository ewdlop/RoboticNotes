import pybullet as p
import time
import pybullet_data
import math

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
#robotId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
robotId = p.loadURDF("test_robot_3.urdf",cubeStartPos, cubeStartOrientation)
mode = p.POSITION_CONTROL
jointIndex = 0 # test different joints, starting at
for i in range (10000):
 p.stepSimulation()
 p.setJointMotorControl2(robotId, jointIndex, controlMode=mode, targetPosition=0.4 + 0.8 * math.sin(i * 0.01 + 0.6))
 time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(cubePos,cubeOrn)
p.disconnect()