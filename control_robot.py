from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

class Robot:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 9))
        self.gs = GridSpec(12, 6, figure=self.fig, hspace=0.5)
        self.ax = self.fig.add_subplot(self.gs[:, :3], projection='3d')
        self.ax.set_title("Tree structure: ABB IRB 4400/60")
        self.ax_det = self.fig.add_subplot(self.gs[:, 3:5])
        self.ax_det.set_title("Jacobian Matrix Determinant")
        self.sliders = []
        self.sliders_cartesian = []
        self.initial_angles = [np.radians(0), np.radians(0), np.radians(0), np.radians(0), np.radians(0), np.radians(0)]
        self.current_joint_angles = self.initial_angles
        self.joint_limits = [
            [-165, 165], # Joint 1
            [-100, 100], # Joint 2 - Adjust with coupling J23
            [-100, 100], # Joint 3 - Adjust with coupling J23
            [-200, 200], # Joint 4
            [-120, 120], # Joint 5
            [-400, 400], # Joint 6
        ]
        self.bounds = (
            [np.radians(limit[0]) for limit in self.joint_limits], 
            [np.radians(limit[1]) for limit in self.joint_limits]
        )

        self.determinants = []

    def rot_z(self, angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle),  np.cos(angle), 0, 0],
            [0,              0,             1, 0],
            [0,              0,             0, 1]
        ])

    def rot_y(self, angle):
        return np.array([
            [ np.cos(angle), 0, np.sin(angle), 0],
            [ 0,             1, 0,             0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [ 0,             0, 0,             1]
        ])

    def rot_x(self, angle):
        return np.array([
            [1, 0,              0,               0],
            [0, np.cos(angle), -np.sin(angle),   0],
            [0, np.sin(angle),  np.cos(angle),   0],
            [0, 0,              0,               1]
        ])

    def trans(self, x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    def plot_frame(self, ax, T, scale=0.1, label=None):
        origin = T[:3, 3]
        x_axis = origin + scale * T[:3, 0]
        y_axis = origin + scale * T[:3, 1]
        z_axis = origin + scale * T[:3, 2]
        
        ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r-', linewidth=2)
        ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g-', linewidth=2)
        ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b-', linewidth=2)
        
        if label:
            ax.text(origin[0], origin[1], origin[2], label, fontsize=10)

    def calculate_tcp(self, theta1, theta2, theta3, theta4, theta5, theta6):
        T1 = self.rot_z(theta1)
        T2 = T1 @ self.trans(0.2, 0, 0.68) @ self.rot_y(theta2)
        T3 = T2 @ self.trans(0, 0, 0.89) @ self.rot_y(theta3-theta2)
        T4 = T3 @ self.trans(0.25, 0, 0.15) @ self.rot_x(theta4)
        T5 = T4 @ self.trans(0.63, 0, 0) @ self.rot_y(theta5)
        T6 = T5 @ self.trans(0.14, 0, 0) @ self.rot_x(theta6)
        T_tool = T6 @ self.trans(0.239, 0, 0) @ self.rot_y(np.pi/2)
        
        position = T_tool[:3, 3]
        orientation = R.from_matrix(T_tool[:3, :3]).as_quat()

        return position, orientation

    def jacobian(self, theta1, theta2, theta3, theta4, theta5, theta6):
        q = [theta1, theta2, theta3, theta4, theta5, theta6]
        h = 1e-6
        J = np.empty((6, 6))
        
        for i in range(6):
            for j in range(6):
                pos, orient = self.calculate_tcp(*q)
                p = np.array([pos[0], pos[1], pos[2], orient[0], orient[1], orient[2], orient[3]])
                q_h = q.copy()
                q_h[j] += h
                pos_h, orient_h = self.calculate_tcp(*q_h)
                ph = np.array([pos_h[0], pos_h[1], pos_h[2], orient_h[0], orient_h[1], orient_h[2], orient_h[3]])
                J[i, j] = (ph[i] - p[i]) / h
        
        return J

    def inverse_kinematics(self, target_pose):
        def objective_function(joint_values):
            pos, orient = self.calculate_tcp(*joint_values)
            dp = target_pose[:3] - pos
            do = target_pose[3:] - orient
            result = [dp[0], dp[1], dp[2], do[0], do[1], do[2], do[3]]
            return np.array(result)
        
        initial_guess = [0, 0, 0, 0, 0, 0]
        result = least_squares(objective_function, initial_guess, method='trf', bounds=self.bounds)

        return result.x

    def setup_controls(self):
        labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$', r'$\theta_5$', r'$\theta_6$']
        labels_cartesian = ["x", "y", "z"]
        cartesian_limits = [
            [-2, -2, 0],
            [2, 2, 2]
        ]
        initial_pos_cartesian, _ = self.calculate_tcp(*self.current_joint_angles)

        slider_title_cartesian = self.fig.add_subplot(self.gs[0, 5], aspect=0.1)
        slider_title_cartesian.set_axis_off()
        slider_title_cartesian.set_title("Cartesian Control")
        slider_title_joint = self.fig.add_subplot(self.gs[4, 5], aspect=0.1)
        slider_title_joint.set_axis_off()
        slider_title_joint.set_title("Joint Control")

        for i in range(3):
            ax_slider = self.fig.add_subplot(self.gs[i + 1, 5])
            ax_slider.set_facecolor('lightgoldenrodyellow')
            slider = Slider(ax_slider, labels_cartesian[i], cartesian_limits[0][i], cartesian_limits[1][i], valinit=initial_pos_cartesian[i])
            slider.on_changed(lambda new_val, axis=labels_cartesian[i]: self.update_cartesian(new_val, axis))
            self.sliders_cartesian.append(slider)

        for i in range(6):
            ax_slider = self.fig.add_subplot(self.gs[i + 5, 5])
            ax_slider.set_facecolor('lightgoldenrodyellow')
            slider = Slider(ax_slider, labels[i], self.joint_limits[i][0], self.joint_limits[i][1], valinit=self.initial_angles[i])
            slider.on_changed(lambda new_val, joint=i: self.update(new_val, joint))
            self.sliders.append(slider)

        reset_ax = self.fig.add_subplot(self.gs[10, 5])
        reset_ax.set_facecolor('lightgoldenrodyellow')
        self.reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
        self.reset_button.on_clicked(self.reset_simulation)

    def plot_robot(self, theta1, theta2, theta3, theta4, theta5, theta6, ax):
        T1 = self.rot_z(theta1)
        T2 = T1 @ self.trans(0.2, 0, 0.68) @ self.rot_y(theta2)
        T3 = T2 @ self.trans(0, 0, 0.89) @ self.rot_y(theta3-theta2)
        T4 = T3 @ self.trans(0.25, 0, 0.15) @ self.rot_x(theta4)
        T5 = T4 @ self.trans(0.63, 0, 0) @ self.rot_y(theta5)
        T6 = T5 @ self.trans(0.14, 0, 0) @ self.rot_x(theta6)
        T_tool = T6 @ self.trans(0.239, 0, 0) @ self.rot_y(np.pi/2)

        frames = [np.eye(4), T1, T2, T3, T4, T5, T6, T_tool]
        labels = ['World', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Tool']

        for T, label in zip(frames, labels):
            self.plot_frame(ax, T, label=label)

        for i in range(len(frames) - 1):
            start = frames[i][:3, 3]
            end = frames[i+1][:3, 3]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k--', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-0.5, 2.5)

    def plot(self):
        self.plot_robot(*[np.radians(angle) for angle in self.initial_angles], self.ax)
        self.setup_controls()
        plt.show()

    def update(self, val, joint):
        self.ax.clear()
        self.ax_det.clear()
        joint_angles = [np.radians(slider.val) for slider in self.sliders]
        self.current_joint_angles = joint_angles
        self.plot_robot(*joint_angles, self.ax)
        J = self.jacobian(*joint_angles)
        det = np.linalg.det(J)
        self.determinants.append(det)
        self.ax_det.plot(self.determinants)
        pos, orient = self.calculate_tcp(*self.current_joint_angles)
        new_pos_string = [f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], [{orient[0]:.2f}, {orient[1]:.2f}, {orient[2]:.2f}, {orient[3]:.2f}]"]
        # print("New Pos: ", new_pos_string)

    def update_cartesian(self, val, axis):
        self.ax.clear()
        self.ax_det.clear()
        pos, orient = self.calculate_tcp(*self.current_joint_angles)
        # print(f"Start Pose: [{pos[0]}, {pos[1]}, {pos[2]}], [{orient[0]}, {orient[1]}, {orient[2]}, {orient[3]}]")
        new_pos = [slider.val for slider in self.sliders_cartesian]
        target_pose = [new_pos[0], new_pos[1], new_pos[2], orient[0], orient[1], orient[2], orient[3]]
        # print("Target Pose: ", [f"{p:.2f}" for p in target_pose])
        new_angles = self.inverse_kinematics(target_pose)
        for i, slider in enumerate(self.sliders):
            slider.set_val(np.degrees(new_angles[i])) 
        self.current_joint_angles = new_angles
        self.plot_robot(*new_angles, self.ax)
        J = self.jacobian(*new_angles)
        det = np.linalg.det(J)
        self.determinants.append(det)
        self.ax_det.plot(self.determinants)

    def reset_simulation(self, event):
        self.ax.clear()
        self.ax_det.clear()
        self.determinants = []
        for slider, angle in zip(self.sliders, self.initial_angles):
            slider.set_val(np.degrees(angle))
        self.current_joint_angles = self.initial_angles
        self.plot_robot(*self.initial_angles, self.ax)
        plt.draw()

def main():
    robot = Robot()
    robot.plot()

if __name__ == "__main__":
    main()
