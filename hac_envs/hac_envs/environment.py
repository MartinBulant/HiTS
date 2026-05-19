import os
import numpy as np
import mujoco
import mujoco.viewer


class MjSimCompat:
    def __init__(self, model, data):
        self.model = model
        self.data = data


class Environment:
    def __init__(
        self,
        model_name,
        goal_space_train,
        goal_space_test,
        project_state_to_end_goal,
        end_goal_thresholds,
        initial_state_space,
        subgoal_bounds,
        project_state_to_subgoal,
        subgoal_thresholds,
        max_actions=1200,
        num_frames_skip=10,
        show=False,
    ):
        self.name = model_name

        # Create MuJoCo Simulation
        model_path = os.path.join(os.path.dirname(__file__), "mujoco_files", model_name)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.sim = MjSimCompat(self.model, self.data)

        # State/action dimensions
        if model_name == "pendulum.xml":
            self.state_dim = 2 * len(self.data.qpos) + len(self.data.qvel)
        else:
            self.state_dim = len(self.data.qpos) + len(self.data.qvel)

        self.action_dim = self.model.nu
        self.action_bounds = self.model.actuator_ctrlrange[:, 1]
        self.action_offset = np.zeros(len(self.action_bounds))
        self.end_goal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_subgoal = project_state_to_subgoal

        self.subgoal_bounds_symmetric = np.zeros(len(self.subgoal_bounds))
        self.subgoal_bounds_offset = np.zeros(len(self.subgoal_bounds))
        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (
                self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]
            ) / 2
            self.subgoal_bounds_offset[i] = (
                self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]
            )

        self.end_goal_thresholds = end_goal_thresholds
        self.subgoal_thresholds = subgoal_thresholds
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = [
            "Magenta",
            "Green",
            "Red",
            "Blue",
            "Cyan",
            "Orange",
            "Maroon",
            "Gray",
            "White",
            "Black",
        ]
        self.max_actions = max_actions

        self.visualize = show
        self.viewer = None
        if self.visualize:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.num_frames_skip = num_frames_skip

    def get_state(self):
        if self.name == "pendulum.xml":
            return np.concatenate(
                [
                    np.cos(self.data.qpos),
                    np.sin(self.data.qpos),
                    self.data.qvel,
                ]
            )
        else:
            return np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)

        for i in range(len(self.data.qpos)):
            self.data.qpos[i] = np.random.uniform(
                self.initial_state_space[i][0], self.initial_state_space[i][1]
            )
        for i in range(len(self.data.qvel)):
            self.data.qvel[i] = np.random.uniform(
                self.initial_state_space[len(self.data.qpos) + i][0],
                self.initial_state_space[len(self.data.qpos) + i][1],
            )

        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def execute_action(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            mujoco.mj_step(self.model, self.data)
            if self.visualize and self.viewer is not None:
                self.viewer.sync()

        return self.get_state()

    def display_end_goal(self, end_goal):
        if self.name == "pendulum.xml":
            self.data.mocap_pos[0] = np.array(
                [0.5 * np.sin(end_goal[0]), 0, 0.5 * np.cos(end_goal[0]) + 0.6]
            )
        elif self.name == "ur5.xml":
            theta_1, theta_2, theta_3 = end_goal[0], end_goal[1], end_goal[2]

            upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            T_1_0 = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
            )
            T_2_1 = np.array(
                [
                    [np.cos(theta_1), -np.sin(theta_1), 0, 0],
                    [np.sin(theta_1), np.cos(theta_1), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            T_3_2 = np.array(
                [
                    [np.cos(theta_2), 0, np.sin(theta_2), 0],
                    [0, 1, 0, 0.13585],
                    [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                    [0, 0, 0, 1],
                ]
            )
            T_4_3 = np.array(
                [
                    [np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                    [0, 1, 0, 0],
                    [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                    [0, 0, 0, 1],
                ]
            )

            joint_pos = [
                T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3],
                T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3],
                T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3],
            ]
            for i in range(3):
                self.data.mocap_pos[i] = joint_pos[i]
        else:
            assert False, "Provide display end goal function in environment.py file"

    def get_next_goal(self, test):
        end_goal = np.zeros(len(self.goal_space_test))

        if self.name == "ur5.xml":
            goal_possible = False
            while not goal_possible:
                end_goal = np.zeros(self.end_goal_dim)
                end_goal[0] = np.random.uniform(
                    self.goal_space_test[0][0], self.goal_space_test[0][1]
                )
                end_goal[1] = np.random.uniform(
                    self.goal_space_test[1][0], self.goal_space_test[1][1]
                )
                end_goal[2] = np.random.uniform(
                    self.goal_space_test[2][0], self.goal_space_test[2][1]
                )

                theta_1, theta_2, theta_3 = end_goal[0], end_goal[1], end_goal[2]

                upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
                forearm_pos_3 = np.array([0.425, 0, 0, 1])
                wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

                T_1_0 = np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
                )
                T_2_1 = np.array(
                    [
                        [np.cos(theta_1), -np.sin(theta_1), 0, 0],
                        [np.sin(theta_1), np.cos(theta_1), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
                T_3_2 = np.array(
                    [
                        [np.cos(theta_2), 0, np.sin(theta_2), 0],
                        [0, 1, 0, 0.13585],
                        [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                        [0, 0, 0, 1],
                    ]
                )
                T_4_3 = np.array(
                    [
                        [np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                        [0, 1, 0, 0],
                        [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                        [0, 0, 0, 1],
                    ]
                )

                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = (
                    T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]
                )

                if (
                    np.absolute(end_goal[0]) > np.pi / 4
                    and forearm_pos[2] > 0.05
                    and wrist_1_pos[2] > 0.15
                ):
                    goal_possible = True

        elif not test and self.goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(
                    self.goal_space_train[i][0], self.goal_space_train[i][1]
                )
        else:
            assert self.goal_space_test is not None, (
                'Need goal space for testing. Set goal_space_test variable in "design_env.py" file'
            )
            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(
                    self.goal_space_test[i][0], self.goal_space_test[i][1]
                )

        self.display_end_goal(end_goal)
        return end_goal

    def display_subgoals(self, subgoals):
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals), 11)):
            if self.name == "pendulum.xml":
                self.data.mocap_pos[i] = np.array(
                    [
                        0.5 * np.sin(subgoals[subgoal_ind][0]),
                        0,
                        0.5 * np.cos(subgoals[subgoal_ind][0]) + 0.6,
                    ]
                )
                self.model.site_rgba[i][3] = 1
                subgoal_ind += 1

            elif self.name == "ur5.xml":
                theta_1 = subgoals[subgoal_ind][0]
                theta_2 = subgoals[subgoal_ind][1]
                theta_3 = subgoals[subgoal_ind][2]

                upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
                forearm_pos_3 = np.array([0.425, 0, 0, 1])
                wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

                T_1_0 = np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
                )
                T_2_1 = np.array(
                    [
                        [np.cos(theta_1), -np.sin(theta_1), 0, 0],
                        [np.sin(theta_1), np.cos(theta_1), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
                T_3_2 = np.array(
                    [
                        [np.cos(theta_2), 0, np.sin(theta_2), 0],
                        [0, 1, 0, 0.13585],
                        [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                        [0, 0, 0, 1],
                    ]
                )
                T_4_3 = np.array(
                    [
                        [np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                        [0, 1, 0, 0],
                        [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                        [0, 0, 0, 1],
                    ]
                )

                joint_pos = [
                    T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3],
                    T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3],
                    T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3],
                ]
                for j in range(3):
                    self.data.mocap_pos[3 + 3 * (i - 1) + j] = np.copy(joint_pos[j])
                    self.model.site_rgba[3 + 3 * (i - 1) + j][3] = 1

                subgoal_ind += 1
            else:
                self.data.mocap_pos[i] = subgoals[subgoal_ind]
                self.model.site_rgba[i][3] = 1
                subgoal_ind += 1
