import cvxpy as cp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class randomTraj:
    def __init__(self):
        self.totalTime = 10
        self.pointNum = 10
        self.startPoint = np.array([0, 0, 0])
        self.endPoint = np.array([100, 100, 100])
        self.stepLength = (self.endPoint[0] - self.startPoint[0]) / self.pointNum
        self.traj = []
        self.snap = 0

    def getRandomPoint(self, index):
        return (index + np.sin(np.random.uniform(-1, 1) * np.pi)) * self.stepLength - 2 * self.stepLength

    def generateTraj(self):
        self.traj.append(self.startPoint)
        for i in range(self.pointNum):
            point = np.array([self.getRandomPoint(i), self.getRandomPoint(i), self.getRandomPoint(i)])
            self.traj.append(point)
        self.traj.append(self.endPoint)

    def computeSnap(self):
        traj = np.array(self.traj)
        snap = np.sum(np.power(np.diff(traj, n=4, axis=0), 2))
        self.snap = snap

    def optimizeTraj(self):
        traj = cp.Variable((self.pointNum + 2, 3))
        objective = cp.Minimize(self.snap)
        constraints = [traj[0, :] == self.startPoint,
                       traj[-1, :] == self.endPoint]
        # 添加对中间点位置的约束
        for i in range(1, self.pointNum + 1):
            constraints.append(traj[i, :] == self.traj[i])
        # 添加对中间点速度的约束
        for i in range(1, self.pointNum + 1):
            velocity = cp.diff(traj[i, :], axis=0)
            constraints.append(cp.norm(velocity, 'inf') <= 20)
        # 添加对中间点加速度的约束
        for i in range(1, self.pointNum + 1):
            velocity = cp.diff(traj[i, :], axis=0)
            acceleration = cp.diff(velocity, axis=0)
            constraints.append(cp.norm(acceleration, 'inf') <= 45)
        # 添加对起点和终点速度的约束
        constraints.append(cp.diff(traj[0, :], axis=0) == 0)
        constraints.append(cp.diff(traj[-1, :], axis=0) == 0)
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.ECOS)
        if result is not None:
            self.traj = traj.value
        else:
            print("Error: The optimization problem could not be solved.")

    def plotTraj(self):
        fig = plt.figure()

        # 创建轨迹的3D子图
        ax0 = fig.add_subplot(311, projection='3d')
        traj = np.array(self.traj)
        ax0.plot(traj[:, 0], traj[:, 1], traj[:, 2])
        ax0.scatter(*self.startPoint, color='red')
        ax0.scatter(*self.endPoint, color='green')
        for i in range(1, len(traj) - 1):
            ax0.scatter(*traj[i], color='blue')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        ax0.set_title('Trajectory')

        # 创建速度的2D子图
        ax1 = fig.add_subplot(312)
        velocity = np.diff(traj, axis=0, prepend=[traj[0, :]])
        velocity_norm = np.linalg.norm(velocity, axis=1)
        ax1.plot(velocity_norm)
        ax1.set_title('Velocity')

        # 创建加速度的2D子图
        ax2 = fig.add_subplot(313)
        acceleration = np.diff(velocity, axis=0, prepend=[velocity[0, :]])
        acceleration_norm = np.linalg.norm(acceleration, axis=1)
        ax2.plot(acceleration_norm)
        ax2.set_title('Acceleration')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    traj = randomTraj()
    traj.generateTraj()
    traj.computeSnap()
    traj.plotTraj()

