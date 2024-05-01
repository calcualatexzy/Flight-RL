import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -9.81)

cylinderStartPos = [-3, 0, 0.5]  # Start position at x = -3
cylinderStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

planeId = p.loadURDF("plane.urdf")
# Specify full path to the URDF file unless included in PyBullet's data path
cylinderId = p.loadURDF("/media/ziyi/MyPassport/zju/Ego-RL/Flight-RL/FlightEnv/assets/cylinder.urdf",
                        cylinderStartPos, cylinderStartOrientation)

if cylinderId < 0:
    print("Error loading cylinder URDF file.")
    exit()

p.changeDynamics(planeId, -1, lateralFriction=0, spinningFriction=0, rollingFriction=0)

# Loop to simulate the motion
while True:
    for i in range(600):  # Adjust the number of iterations as needed
        # Calculate the new position of the cylinder
        x = -3 + i * (6 / 600)  # Linear interpolation from -3 to 3
        new_pos = [x, 0, 0.5]
        # Set the position of the cylinder
        p.resetBasePositionAndOrientation(cylinderId, new_pos, cylinderStartOrientation)
        p.stepSimulation()
        time.sleep(1. / 240.)
    for i in range(600):  # Adjust the number of iterations as needed
        # Calculate the new position of the cylinder
        x = 3 - i * (6 / 600)  # Linear interpolation from -3 to 3
        new_pos = [x, 0, 0.5]
        # Set the position of the cylinder
        p.resetBasePositionAndOrientation(cylinderId, new_pos, cylinderStartOrientation)
        p.stepSimulation()
        time.sleep(1. / 240.)


p.disconnect()
