import pybullet as p
import time
import pybullet_data
import math

def setup_simulation():
    """Initialize PyBullet simulation environment"""
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    return physicsClient, planeId

def load_robot():
    """Load the humanoid robot from MJCF file"""
    try:
        robotIds = p.loadMJCF("ant.xml")
        print(f"MJCF loaded successfully. Number of bodies: {len(robotIds)}")
        
        # The humanoid is typically the second body (index 1)
        if len(robotIds) > 1:
            robotId = robotIds[1]
        else:
            robotId = robotIds[0]
            
        print(f"Using robotId: {robotId}")
        
        # Check if the robot was loaded properly
        basePos, baseOrn = p.getBasePositionAndOrientation(robotId)
        print(f"Robot base position: {basePos}")
        return robotId
        
    except Exception as e:
        print(f"Failed to load MJCF: {e}")
        # Fallback to a built-in humanoid robot
        robotId = p.loadURDF("humanoid/humanoid.urdf")
        print("Loaded humanoid.urdf as fallback")
        return robotId

def analyze_joints(robotId):
    """Analyze and categorize all joints in the robot"""
    numJoints = p.getNumJoints(robotId)
    print(f"\nNumber of joints in the robot: {numJoints}")
    print("="*80)
    
    controllable_joints = []
    joint_info = {}
    
    for i in range(numJoints):
        jointInfo = p.getJointInfo(robotId, i)
        jointName = jointInfo[1].decode('utf-8')
        jointType = jointInfo[2]
        jointLowerLimit = jointInfo[8]
        jointUpperLimit = jointInfo[9]
        jointMaxForce = jointInfo[10]
        jointMaxVelocity = jointInfo[11]
        
        joint_types = {
            0: "REVOLUTE",
            1: "PRISMATIC", 
            2: "SPHERICAL",
            3: "PLANAR",
            4: "FIXED"
        }
        
        type_name = joint_types.get(jointType, f"UNKNOWN({jointType})")
        
        print(f"Joint {i:2d}: '{jointName:15s}' Type: {type_name:10s} "
              f"Limits: [{jointLowerLimit:6.1f}, {jointUpperLimit:6.1f}] "
              f"MaxForce: {jointMaxForce:6.1f}")
        
        # Store joint information
        joint_info[i] = {
            'name': jointName,
            'type': jointType,
            'lower_limit': jointLowerLimit,
            'upper_limit': jointUpperLimit,
            'max_force': jointMaxForce
        }
        
        # Skip free joints (type 0 if it's the root) and fixed joints (type 4)
        # But include revolute joints (type 0 when they're not root joints)
        if jointType in [0, 1]:  # REVOLUTE or PRISMATIC
            # Additional check: joints with meaningful limits are usually controllable
            if jointLowerLimit < jointUpperLimit:
                controllable_joints.append(i)
                print(f"  -> CONTROLLABLE")
    
    print(f"\nControllable joints: {controllable_joints}")
    return controllable_joints, joint_info

def create_joint_control_pattern(controllable_joints, joint_info):
    """Create different control patterns for different joint groups"""
    patterns = {}
    
    for joint_idx in controllable_joints:
        joint_name = joint_info[joint_idx]['name'].lower()
        
        # Categorize joints by body part
        if 'hip' in joint_name:
            patterns[joint_idx] = 'hip'
        elif 'knee' in joint_name:
            patterns[joint_idx] = 'knee'
        elif 'shoulder' in joint_name:
            patterns[joint_idx] = 'shoulder'
        elif 'elbow' in joint_name:
            patterns[joint_idx] = 'elbow'
        elif 'abdomen' in joint_name:
            patterns[joint_idx] = 'abdomen'
        else:
            patterns[joint_idx] = 'other'
    
    return patterns

def control_robot(robotId, controllable_joints, joint_info, joint_patterns):
    """Main control loop for the robot"""
    print(f"\nStarting robot control with {len(controllable_joints)} controllable joints...")
    
    for step in range(2000):
        p.stepSimulation()
        
        # Different control patterns for different joint types
        for joint_idx in controllable_joints:
            joint_name = joint_info[joint_idx]['name']
            joint_pattern = joint_patterns.get(joint_idx, 'other')
            lower_limit = joint_info[joint_idx]['lower_limit']
            upper_limit = joint_info[joint_idx]['upper_limit']
            
            # Calculate target position based on joint type and time
            time_factor = step * 0.02
            
            if joint_pattern == 'hip':
                # Hip joints - walking motion
                if 'right' in joint_name.lower():
                    target = 0.3 * math.sin(time_factor)
                else:
                    target = 0.3 * math.sin(time_factor + math.pi)
            elif joint_pattern == 'knee':
                # Knee joints - bending motion
                target = -0.5 + 0.3 * abs(math.sin(time_factor * 2))
            elif joint_pattern == 'shoulder':
                # Shoulder joints - arm swinging
                target = 0.4 * math.sin(time_factor * 0.5)
            elif joint_pattern == 'elbow':
                # Elbow joints - arm bending
                target = -0.3 + 0.2 * math.sin(time_factor)
            elif joint_pattern == 'abdomen':
                # Abdomen joints - torso movement
                target = 0.1 * math.sin(time_factor * 0.3)
            else:
                # Default pattern for other joints
                target = 0.2 * math.sin(time_factor + joint_idx)
            
            # Clamp target to joint limits
            target = max(lower_limit, min(upper_limit, target))
            
            # Apply control using setJointMotorControl2
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=100,  # Adjust force as needed
                maxVelocity=1.0  # Limit velocity for smooth motion
            )
        
        time.sleep(1. / 240.)
        
        # Print status every 200 steps
        if step % 200 == 0:
            basePos, baseOrn = p.getBasePositionAndOrientation(robotId)
            print(f"Step {step:4d}: Robot position {basePos[0]:.2f}, {basePos[1]:.2f}, {basePos[2]:.2f}")

def main():
    """Main function to run the simulation"""
    # Setup simulation
    physicsClient, planeId = setup_simulation()
    
    # Load robot
    robotId = load_robot()
    
    # Analyze joints
    controllable_joints, joint_info = analyze_joints(robotId)
    
    if not controllable_joints:
        print("No controllable joints found! Just running basic simulation...")
        for i in range(1000):
            p.stepSimulation()
            time.sleep(1. / 240.)
    else:
        # Create control patterns
        joint_patterns = create_joint_control_pattern(controllable_joints, joint_info)
        
        print(f"\nJoint control patterns:")
        for joint_idx, pattern in joint_patterns.items():
            joint_name = joint_info[joint_idx]['name']
            print(f"  Joint {joint_idx:2d} ({joint_name:15s}): {pattern}")
        
        # Control the robot
        control_robot(robotId, controllable_joints, joint_info, joint_patterns)
    
    # Final status
    finalPos, finalOrn = p.getBasePositionAndOrientation(robotId)
    print(f"\nFinal position: {finalPos}, orientation: {finalOrn}")
    
    # Keep simulation running for a bit to observe final state
    print("Simulation complete. Keeping window open for 5 seconds...")
    time.sleep(5)
    
    p.disconnect()

if __name__ == "__main__":
    main()