import pybullet as p
import time
import pybullet_data
import math

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Try loading the MJCF file with better error handling
try:
    robotIds = p.loadMJCF("ant.xml")
    print(f"MJCF loaded successfully. Number of bodies: {len(robotIds)}")
    robotId = robotIds[1]
    print(f"Using robotId: {robotId}")
    
    # Check if the robot was loaded properly
    basePos, baseOrn = p.getBasePositionAndOrientation(robotId)
    print(f"Robot base position: {basePos}")
    
except Exception as e:
    print(f"Failed to load MJCF: {e}")
    # Fallback to a built-in robot
    robotId = p.loadURDF("r2d2.urdf")
    print("Loaded R2D2 as fallback")

# Get the number of joints in the robot
numJoints = p.getNumJoints(robotId)
print(f"Number of joints in the robot: {numJoints}")

# Print ALL joint info for debugging
for i in range(numJoints):
    jointInfo = p.getJointInfo(robotId, i)
    jointName = jointInfo[1].decode('utf-8')
    jointType = jointInfo[2]
    jointLowerLimit = jointInfo[8]
    jointUpperLimit = jointInfo[9]
    jointMaxForce = jointInfo[10]
    jointMaxVelocity = jointInfo[11]
    
    print(f"Joint {i}: Name='{jointName}', Type={jointType}, "
          f"Limits=[{jointLowerLimit}, {jointUpperLimit}], "
          f"MaxForce={jointMaxForce}, MaxVel={jointMaxVelocity}")

# Try to identify joints by name that should be controllable
controllable_joints = []
if numJoints > 0:
    for i in range(numJoints):
        jointInfo = p.getJointInfo(robotId, i)
        jointType = jointInfo[2]
        jointName = jointInfo[1].decode('utf-8')
        print(f"Checking joint {i}: Name='{jointName}', Type={jointType}")

        controllable_joints.append(i)

        # Skip free joints (type 0) and fixed joints (type 4)
        if jointType not in [0, 4]:
            controllable_joints.append(i)
            print(f"Found controllable joint {i}: {jointName} (type {jointType})")

print(f"Controllable joints: {controllable_joints}")

if controllable_joints:
    # Try controlling the first few controllable joints
    mode = p.POSITION_CONTROL
    for step in range(1000):
        p.stepSimulation()
        
        target = 0.2 * math.sin(step * 0.01 + 10 * 1.0)
        p.setJointMotorControl2(robotId, 1, controlMode=mode, 
                              targetPosition=target, force=50)

        # # Control multiple joints if available
        # for idx, jointIndex in enumerate(controllable_joints[:4]):  # Control up to 3 joints
        #     target = 0.2 * math.sin(step * 0.01 + idx * 1.0)
        #     p.setJointMotorControl2(robotId, jointIndex, controlMode=mode, 
        #                           targetPosition=target, force=50)
        
        time.sleep(1. / 240.)
        
        # Print position every 100 steps
        if step % 100 == 0:
            basePos, baseOrn = p.getBasePositionAndOrientation(robotId)
            print(f"Step {step}: Robot position {basePos}")
else:
    print("No controllable joints found! Just running simulation...")
    for i in range(1000):
        p.stepSimulation()
        time.sleep(1. / 240.)

cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
print(f"Final position: {cubePos}, orientation: {cubeOrn}")
p.disconnect()