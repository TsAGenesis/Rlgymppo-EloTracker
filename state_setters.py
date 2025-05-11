import math
import random
from collections import namedtuple
from typing import List, Union

import numpy as np
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, \
    CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, BOOST_LOCATIONS
from rlgym_sim.utils.gamestates import PhysicsObject
from rlgym_sim.utils.math import rand_vec3
from rlgym_sim.utils.state_setters.random_state import RandomState
from rlgym_sim.utils.state_setters import StateWrapper, StateSetter
from rlgym_sim.utils.state_setters.wrappers import CarWrapper

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi

DEG_TO_RAD = np.pi / 180

from numpy import random as rand

X_MAX = 7000
Y_MAX = 9000
Z_MAX_CAR = 1900
GOAL_HEIGHT = 642.775
PITCH_MAX = np.pi / 2
ROLL_MAX = np.pi
BLUE_GOAL_POSITION = np.asarray([0, -5120, 0])
ORANGE_GOAL_POSITION = np.asarray([0, 5120, 0])



class DefaultState(StateSetter):
    SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                      0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                        [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                        np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        rand_default_or_diff = random.randint(0, 3)
        # rand_default_or_diff = 0
        # DEFAULT KICKOFFS POSITIONS
        if rand_default_or_diff == 0:

            # possible kickoff indices are shuffled
            spawn_inds = [0, 1, 2, 3, 4]
            random.shuffle(spawn_inds)

            blue_count = 0
            orange_count = 0
            for car in state_wrapper.cars:
                pos = [0, 0, 0]
                yaw = 0
                # team_num = 0 = blue team
                if car.team_num == 0:
                    # select a unique spawn state from pre-determined values
                    pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                    yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                    blue_count += 1
                # team_num = 1 = orange team
                elif car.team_num == 1:
                    # select a unique spawn state from pre-determined values
                    pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                    yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                    orange_count += 1
                # set car state values
                car.set_pos(*pos)
                car.set_rot(yaw=yaw)
                car.boost = 0.33
        else:
            """SPAWN POS RANGE
            Y = CENTER BACK KICKOFF [-4608, 0], [4608, 0]; BACK LEFT/RIGHT KICKOFFS [-3000, 0], [3000,0]"""

            # print("Advanced Kickoffs")
            # SPAWN POSITION
            spawn_inds = [0, 1, 2, 3, 4]
            spawn_pos = np.random.choice(spawn_inds)

            same_pos = random.randint(0, 1)
            # print(f"Same position = {same_pos}")

            # RIGHT CORNER
            if spawn_pos == 0:
                # print("Right Corner")
                # SAME POS
                if same_pos == 1:
                    slope = -2560 / -2048
                    posX = np.random.uniform(-2048, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 200)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (abs(posX), abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -2560 / -2048
                        blue_posX = np.random.uniform(-2048, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        slope = 2560 / 2048
                        orange_posX = np.random.uniform(2048, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            # LEFT CORNER
            elif spawn_pos == 1:
                # print("Left Corner")
                # SAME POS
                if same_pos == 1:
                    slope = -2560 / 2048
                    posX = np.random.uniform(2048, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (-abs(posX), abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -2560 / 2048
                        blue_posX = np.random.uniform(2048, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        slope = 2560 / -2048
                        orange_posX = np.random.uniform(-2048, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        #  print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 2:
                # print("Back Right")
                # SAME POS
                if same_pos == 1:
                    slope = -3840 / -256
                    posX = np.random.uniform(-256, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)

                    if posY < -3000:
                        blue_spawn_pos = (posX, posY, posZ)
                        orange_spawn_pos = (abs(posX), abs(posY), posZ)
                    else:
                        blue_spawn_pos = (0, posY, posZ)
                        orange_spawn_pos = (0, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -3840 / -256
                        blue_posX = np.random.uniform(-256, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        if blue_posY < -3000:
                            blue_spawn_pos = (blue_posX, blue_posY, posZ)
                        else:
                            blue_spawn_pos = (0, blue_posY, posZ)

                        slope = 3840 / 256
                        orange_posX = np.random.uniform(256, 0)
                        orange_posY = slope * orange_posX
                        if orange_posX > 3000:
                            orange_spawn_pos = (orange_posX, orange_posY, posZ)
                        else:
                            orange_spawn_pos = (0, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 3:
                # print("Back Left")
                # SAME POS
                if same_pos == 1:
                    slope = -3840 / 256
                    posX = np.random.uniform(256, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)

                    if posY < -3000:
                        blue_spawn_pos = (posX, posY, posZ)
                        orange_spawn_pos = (-abs(posX), abs(posY), posZ)
                    else:
                        blue_spawn_pos = (posX, 0, 17)
                        orange_spawn_pos = (0, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -3840 / 256
                        blue_posX = np.random.uniform(256, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        if blue_posY < -3000:
                            blue_spawn_pos = (blue_posX, blue_posY, posZ)
                        else:
                            blue_spawn_pos = (0, blue_posY, posZ)

                        slope = 3840 / -256
                        orange_posX = np.random.uniform(-256, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        if orange_posX > 3000:
                            orange_spawn_pos = (orange_posX, orange_posY, posZ)
                        else:
                            orange_spawn_pos = (0, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        #  print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 4:
                # print("Far Back Center")
                # SAME POS
                if same_pos == 1:
                    posX = 0
                    posY = np.random.uniform(-4608, 0)
                    posZ = np.random.uniform(17, 500)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (posX, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        blue_posX = 0
                        blue_posY = np.random.uniform(-4608, 0)
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        orange_posX = 0
                        orange_posY = np.random.uniform(4608, 0)
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)

all_states = [DefaultState()]


# DynamicScoredReplaySetter(
#         "../replays/states_scores_duels.npz",
#         "../replays/states_scores_doubles.npz",
#         "../replays/states_scores_standard.npz"
#     )

class ProbabilisticStateSetter(StateSetter):

    def __init__(self, states, probs, verbose: int = 1, training: bool = True):
        self.old_state = None
        self.verbose = verbose
        self.training = training
        self.states, self.probs = states, probs

    def reset(self, state_wrapper: StateWrapper):

        if self.training:
            selected_state = random.choices(self.states, weights=self.probs, k=1)[0]
        else:
            selected_state = all_states.pop(0)

        self.old_state = selected_state

        # while not _check_positions(state_wrapper):
        selected_state.reset(state_wrapper)
