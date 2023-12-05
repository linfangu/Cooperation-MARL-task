import gym
from gym.spaces import Discrete, MultiDiscrete, Box, MultiBinary, Dict
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentSync_partobs(MultiAgentEnv):
    def __init__(self, config=None):
        """Config takes in width, height, and ts"""
        config = config or {}
        # Dimensions of the grid.
        self.randomize = config.get("randomize", True)
        self.width = config.get("width", 8)
        self.height = config.get("height", 6)
        self.vision = config.get("Vision", [1, 3])
        self.poke_coords1 = config.get("Poke1", [0, 2])
        self.poke_coords2 = config.get("Poke2", [0, 6])
        self.water_coords1 = config.get("Water1", [5, 2])
        self.water_coords2 = config.get("Water2", [5, 6])
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 200)
        self.movement_reward = config.get("movement_reward", -0.1)
        self.sync_limit = config.get(
            "sync_limit", 2
        )  # default 2 steps but record up to 5
        self.observation_space = Dict(
            {
                "nosepoke": Discrete(self.width * self.height / 2 + 1),
                "water": Discrete(self.width * self.height / 2 + 1),
                "otheragent": Discrete(self.width * self.height / 2 + 1),
                # 0 is unknown location, if known, block 1 to the last
                "self": Discrete(self.width * self.height / 2 + 1),
                "otherpoke": Discrete(self.width * self.height / 2 + 1)
                # "noise" : Discrete(self.width*self.height),
            }
        )
        # 0=up, 1=right, 2=down, 3=left, 4 = nothing
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        self.agent1_pos = [
            random.randint(0, self.height - 1),
            random.randint(0, int(self.width / 2) - 1),
        ]
        self.agent2_pos = [
            random.randint(0, self.height - 1),
            random.randint(int(self.width / 2), self.width - 1),
        ]
        if self.randomize:
            self.water_coords1 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.poke_coords1 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(0, int(self.width / 2) - 1),
            ]
            self.water_coords2 = [
                random.randint(int(self.height / 2), self.height - 1),
                random.randint(int(self.width / 2), self.width - 1),
            ]
            self.poke_coords2 = [
                random.randint(0, int(self.height / 2) - 2),
                random.randint(int(self.width / 2), self.width - 1),
            ]
        # self.agent1_pos = [0,2]
        # self.agent2_pos = [0,5]
        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # reward availability now
        self.water_available1 = 0
        self.water_available2 = 0
        # self.gotwater1 = False
        # self.gotwater2 = False

        # info dict
        self.ncorrect = 0
        self.ncorrect1 = 0
        self.ncorrect2 = 0
        self.nmiss1 = 0
        self.nmiss2 = 0
        self.npoke1 = 0
        self.npoke2 = 0
        self.ndrink1 = 0
        self.ndrink2 = 0

        self.sync_poke = 0
        self.miss = 0
        self.poke_history1 = np.repeat(False, 5)  # poke in the last time point
        self.poke_history2 = np.repeat(False, 5)

        self.freeze1 = 0
        self.freeze2 = 0
        self.timeout = False
        # Reset agent1's visited fields.
        # self.agent1_visited_fields = set([tuple(self.agent1_pos)])

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        self.sync_poke = 0
        self.miss = 0
        # self.gotwater1 = False
        # self.gotwater2 = False
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # time of drinking is random
        if self.freeze1 > 0:
            self.freeze1 = self.freeze1 - 1.0  # no action
            r1 = 0
            events1 = []
        else:
            events1 = self._move(self.agent1_pos, action["agent1"], is_agent1=1)

            # if self.timeout:
            #     events1 = []
            if "drink" in events1:
                self.ndrink1 += 1
            elif "poke" in events1:
                self.npoke1 += 1
            if self.water_available1 == 1.0 and "drink" in events1:
                r1 = 2.0
                # self.gotwater1 = True
                # self.ncorrect1 +=1
                self.freeze1 = np.random.randint(4)
                self.water_available1 = 0
                if self.randomize:
                    self.water_coords1 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
                    self.poke_coords1 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(0, int(self.width / 2) - 1),
                    ]  # upper left corner
            # elif "no_action" in events1:
            #     r1 = -0.1
            else:
                r1 = self.movement_reward

        if self.freeze2 > 0:
            self.freeze2 = self.freeze2 - 1.0  # no action
            r2 = 0
            events2 = []
        else:
            events2 = self._move(self.agent2_pos, action["agent2"], is_agent1=0)

            # if self.timeout:
            #     events2 = []
            if "drink" in events2:
                self.ndrink2 += 1
            elif "poke" in events2:
                self.npoke2 += 1
            if self.water_available2 == 1.0 and "drink" in events2:
                r2 = 2.0
                # self.gotwater2 = True
                # self.ncorrect2 +=1
                self.freeze2 = np.random.randint(4)
                self.water_available2 = 0
                if self.randomize:
                    self.water_coords2 = [
                        random.randint(int(self.height / 2), self.height - 1),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]
                    self.poke_coords2 = [
                        random.randint(0, int(self.height / 2) - 2),
                        random.randint(int(self.width / 2), self.width - 1),
                    ]
            # elif "no_action" in events2:
            #     r2 = -0.1
            else:
                r2 = self.movement_reward
        self.timeout = False
        poke_mat = np.array(
            [
                np.append(self.poke_history1[-self.sync_limit :], "poke" in events1),
                np.append(self.poke_history2[-self.sync_limit :], "poke" in events2),
            ]
        )
        # print(poke_mat)
        if np.all(np.any(poke_mat, axis=1)):  # sync poke
            self.sync_poke = 1
            self.ncorrect += 1
            # reset poking history
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
        # if not in sync, give pushnishment to both and create time out - any poking in the next time step does not count
        elif np.any(poke_mat[:, 0]):
            r1 = -0.5
            r2 = -0.5
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
            self.timeout = True
            self.miss = 1
            if poke_mat[0, 0]:
                self.nmiss1 += 1
            elif poke_mat[1, 0]:
                self.nmiss2 += 1
        else:
            self.poke_history1 = np.append(self.poke_history1[1:], "poke" in events1)
            self.poke_history2 = np.append(self.poke_history2[1:], "poke" in events2)

        if (
            self.sync_poke == 1
            and self.water_available1 == 0
            and self.water_available2 == 0
        ):  # cannot get double reward for sync poke
            r1 = 2.0
            r2 = 2.0
        if self.sync_poke == 1:
            self.water_available1 = 1.0
            self.water_available2 = 1.0

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }
        self.events = {
            "agent1": events1,
            "agent2": events2,
        }
        info = {
            "agent1": {
                "ncorrect": self.ncorrect,
                "ncorrect1": self.ncorrect1,
                "nmiss": self.nmiss1,
                "npoke1": self.npoke1,
                "ndrink1": self.ndrink1,
            },
            "agent2": {
                "ncorrect": self.ncorrect,
                "ncorrect2": self.ncorrect2,
                "nmiss": self.nmiss2,
                "npoke2": self.npoke2,
                "ndrink2": self.ndrink2,
            },
        }

        return obs, rewards, dones, info  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs space for one agent using each
        agent's current x/y-positions.
        """

        # discrete coordinate of the locations of nose poke water port and
        nosepoke_pos1 = int(
            self.poke_coords1[0] * self.width / 2
            + (self.poke_coords1[1] % (self.width / 2) + 1)
        )  # -1
        nosepoke_pos2 = int(
            self.poke_coords2[0] * self.width / 2
            + (self.poke_coords2[1] % (self.width / 2) + 1)
        )  # -1
        water_pos1 = int(
            self.water_coords1[0] * self.width / 2
            + (self.water_coords1[1] % (self.width / 2) + 1)
        )  # -1
        water_pos2 = int(
            self.water_coords2[0] * self.width / 2
            + (self.water_coords2[1] % (self.width / 2) + 1)
        )  # -1
        ag1_discrete_pos = int(
            self.agent1_pos[0] * self.width / 2
            + (self.agent1_pos[1] % (self.width / 2) + 1)
        )  # -1
        ag2_discrete_pos = int(
            self.agent2_pos[0] * self.width / 2
            + (self.agent2_pos[1] % (self.width / 2) + 1)
        )  # -1

        # distance between the two
        # mirror image ( both animals start with lane 0 )
        visible = False  # Initialize visible to False by default

        if (
            abs(self.agent1_pos[0] - self.agent2_pos[0]) <= self.vision[0]
            and abs(self.agent1_pos[1] + self.agent2_pos[1] - self.width / 2)
            <= self.vision[1] - 1
        ):
            visible = True

        # Use more descriptive variable names
        ag1other = ag2_discrete_pos if visible else 0
        ag2other = ag1_discrete_pos if visible else 0
        # noise1 = random.randint(1, self.width*self.height)
        # noise2 = random.randint(1, self.width*self.height)

        return {
            "agent1": {
                "nosepoke": nosepoke_pos1,
                "water": water_pos1,
                "self": ag1_discrete_pos,
                "otheragent": ag1other,
                "otherpoke": nosepoke_pos2,
            },
            "agent2": {
                "nosepoke": nosepoke_pos2,
                "water": water_pos2,
                "self": ag2_discrete_pos,
                "otheragent": ag2other,
                "otherpoke": nosepoke_pos1,
            },
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # check walls.
        if is_agent1:
            bound = self.water_coords1
        else:
            bound = self.water_coords2

        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
        # elif coords[0] >= bound[0]:
        #     coords[0] = bound[0]
        if is_agent1:
            if coords[1] < 0:
                coords[1] = 0
            elif coords[1] >= int(self.width / 2):  # midline
                coords[1] = int(self.width / 2) - 1
        else:
            if coords[1] < self.width / 2:
                coords[1] = int(self.width / 2)
            elif coords[1] >= self.width:
                coords[1] = self.width - 1

        # update agent location
        # nose poke
        if coords == self.poke_coords1 or coords == self.poke_coords2:
            return {"poke"}

        # drink water
        elif coords == self.water_coords1 or coords == self.water_coords2:
            return {"drink"}

        # no action
        elif action == 4:
            return {"no_action"}

        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                # if self.agent1_pos == [r, c]: # flip agent 1's image
                if self.agent1_pos == [r, self.width / 2 - 1 - c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif self.poke_coords1 == [r, self.width / 2 - 1 - c]:
                    print("*", end="")
                elif self.poke_coords2 == [r, c]:
                    print("*", end="")
                elif self.water_coords1 == [r, self.width / 2 - 1 - c]:
                    print(".", end="")
                elif self.water_coords2 == [r, c]:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(
            f"{'!SyncPoke!' if self.sync_poke == 1 else '!Miss!' if self.miss == 1 else ''}"
        )
        print(
            f"{'!Agent1Poke!' if ('poke' in self.events['agent1']) else '!Agent1Drink!' if 'drink' in self.events['agent1'] else ''}"
        )
        print(
            f"{'!Agent2Poke!' if 'poke' in self.events['agent2'] else '!Agent2Drink!' if 'drink' in self.events['agent2'] else ''}"
        )

        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()

        ##############################################


############ single task ###################################


class MultiAgentSing_partobs(MultiAgentEnv):
    def __init__(self, config=None):
        """Config takes in width, height, and ts"""
        config = config or {}
        # Dimensions of the grid.
        self.width = config.get("width", 8)
        self.height = config.get("height", 6)
        self.vision = config.get("Vision", [1, 3])
        self.poke_coords1 = config.get("Poke1", [0, 2])
        self.poke_coords2 = config.get("Poke2", [0, 6])
        self.water_coords1 = config.get("Water1", [5, 2])
        self.water_coords2 = config.get("Water2", [5, 6])
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 200)
        self.movement_reward = config.get("movement_reward", -0.1)
        self.sync_limit = config.get(
            "sync_limit", 2
        )  # default 2 steps but record up to 5
        self.observation_space = Dict(
            {
                "nosepoke": Discrete(self.width * self.height / 2 + 1),
                "water": Discrete(self.width * self.height / 2 + 1),
                "otheragent": Discrete(self.width * self.height / 2 + 1),
                # 0 is unknown location, if known, block 1 to the last
                "self": Discrete(self.width * self.height / 2 + 1),
                "otherpoke": Discrete(self.width * self.height / 2 + 1)
                # "noise" : Discrete(self.width*self.height),
            }
        )
        # 0=up, 1=right, 2=down, 3=left, 4 = nothing
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        self.agent1_pos = [
            random.randint(0, self.height - 1),
            random.randint(0, int(self.width / 2) - 1),
        ]
        self.agent2_pos = [
            random.randint(0, self.height - 1),
            random.randint(int(self.width / 2), self.width - 1),
        ]
        # self.agent1_pos = [0,2]
        # self.agent2_pos = [0,5]
        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0
        self.water_coords1 = [
            random.randint(int(self.height / 2), self.height - 1),
            random.randint(0, int(self.width / 2) - 1),
        ]
        self.poke_coords1 = [
            random.randint(0, int(self.height / 2) - 2),
            random.randint(0, int(self.width / 2) - 1),
        ]
        self.water_coords2 = [
            random.randint(int(self.height / 2), self.height - 1),
            random.randint(int(self.width / 2), self.width - 1),
        ]
        self.poke_coords2 = [
            random.randint(0, int(self.height / 2) - 2),
            random.randint(int(self.width / 2), self.width - 1),
        ]
        # reward availability now
        self.water_available1 = False
        self.water_available2 = False
        # self.gotwater1 = False
        # self.gotwater2 = False

        # info dict
        self.ncorrect = 0
        self.ncorrect1 = 0
        self.ncorrect2 = 0
        self.nmiss1 = 0
        self.nmiss2 = 0
        self.npoke1 = 0
        self.npoke2 = 0
        self.ndrink1 = 0
        self.ndrink2 = 0

        self.poke_history1 = np.array([False, False])  # poke in the last time point
        self.poke_history2 = np.array([False, False])
        self.freeze1 = 0
        self.freeze2 = 0
        self.timeout = False
        # Reset agent1's visited fields.
        # self.agent1_visited_fields = set([tuple(self.agent1_pos)])

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        self.sync_poke = 0
        self.miss = 0
        # self.gotwater1 = False
        # self.gotwater2 = False
        # print(self.water_available1)
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # time of drinking is random
        if self.freeze1 > 0:
            self.freeze1 = self.freeze1 - 1.0  # no action
            r1 = 0
            events1 = []
        else:
            events1 = self._move(self.agent1_pos, action["agent1"], is_agent1=1)

            # if self.timeout:
            #     events1 = []
            if "drink" in events1:
                self.ndrink1 += 1
            elif "poke" in events1:
                self.npoke1 += 1
            if self.water_available1 and "drink" in events1:
                r1 = 2.0
                self.ncorrect1 += 1
                # self.gotwater1 = True
                self.freeze1 = np.random.randint(4)
                self.water_available1 = False
                self.water_coords1 = [
                    random.randint(int(self.height / 2), self.height - 1),
                    random.randint(0, int(self.width / 2) - 1),
                ]  # upper left corner
                self.poke_coords1 = [
                    random.randint(0, int(self.height / 2) - 2),
                    random.randint(0, int(self.width / 2) - 1),
                ]  # upper left corner
            elif (not self.water_available1) and "poke" in events1:
                r1 = 2.0
                self.water_available1 = True
            # elif "no_action" in events1:
            #     r1 = -0.1
            else:
                r1 = self.movement_reward

        if self.freeze2 > 0:
            self.freeze2 = self.freeze2 - 1.0  # no action
            r2 = 0
            events2 = []
        else:
            events2 = self._move(self.agent2_pos, action["agent2"], is_agent1=0)

            # if self.timeout:
            #     events2 = []
            if "drink" in events2:
                self.ndrink2 += 1
            elif "poke" in events2:
                self.npoke2 += 1
            if self.water_available2 and "drink" in events2:
                r2 = 2.0
                self.ncorrect2 += 1
                # self.gotwater2 = True
                self.freeze2 = np.random.randint(4)
                self.water_available2 = False
                self.water_coords2 = [
                    random.randint(int(self.height / 2), self.height - 1),
                    random.randint(int(self.width / 2), self.width - 1),
                ]  # upper left corner
                self.poke_coords2 = [
                    random.randint(0, int(self.height / 2) - 2),
                    random.randint(int(self.width / 2), self.width - 1),
                ]  # upper left corner
            elif (not self.water_available2) and "poke" in events2:
                r2 = 2.0
                self.water_available2 = True
            # elif "no_action" in events2:
            #     r2 = -0.1
            else:
                r2 = self.movement_reward

        poke_mat = np.array(
            [
                np.append(self.poke_history1[-self.sync_limit :], "poke" in events1),
                np.append(self.poke_history2[-self.sync_limit :], "poke" in events2),
            ]
        )
        # print(poke_mat)
        if np.all(np.any(poke_mat, axis=1)):  # sync poke
            self.sync_poke = 1
            self.ncorrect += 1
            # reset poking history
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
        # if not in sync, give pushnishment to both and create time out - any poking in the next time step does not count
        elif np.any(poke_mat[:, 0]):
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
            self.miss = 1
            if poke_mat[0, 0]:
                self.nmiss1 += 1
            elif poke_mat[1, 0]:
                self.nmiss2 += 1
        else:
            self.poke_history1 = np.append(self.poke_history1[1:], "poke" in events1)
            self.poke_history2 = np.append(self.poke_history2[1:], "poke" in events2)

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }
        self.events = {
            "agent1": events1,
            "agent2": events2,
        }

        info = {
            "agent1": {
                "ncorrect": self.ncorrect,
                "ncorrect1": self.ncorrect1,
                "nmiss": self.nmiss1,
                "npoke1": self.npoke1,
                "ndrink1": self.ndrink1,
            },
            "agent2": {
                "ncorrect": self.ncorrect,
                "ncorrect2": self.ncorrect2,
                "nmiss": self.nmiss2,
                "npoke2": self.npoke2,
                "ndrink2": self.ndrink2,
            },
        }
        return obs, rewards, dones, info  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs space for one agent using each
        agent's current x/y-positions.
        """

        # discrete coordinate of the locations of nose poke water port and
        nosepoke_pos1 = int(
            self.poke_coords1[0] * self.width / 2
            + (self.poke_coords1[1] % (self.width / 2) + 1)
        )
        nosepoke_pos2 = int(
            self.poke_coords2[0] * self.width / 2
            + (self.poke_coords2[1] % (self.width / 2) + 1)
        )
        water_pos1 = int(
            self.water_coords1[0] * self.width / 2
            + (self.water_coords1[1] % (self.width / 2) + 1)
        )
        water_pos2 = int(
            self.water_coords2[0] * self.width / 2
            + (self.water_coords2[1] % (self.width / 2) + 1)
        )
        ag1_discrete_pos = int(
            self.agent1_pos[0] * self.width / 2
            + (self.agent1_pos[1] % (self.width / 2) + 1)
        )
        ag2_discrete_pos = int(
            self.agent2_pos[0] * self.width / 2
            + (self.agent2_pos[1] % (self.width / 2) + 1)
        )

        visible = False  # Initialize visible to False by default

        if (
            abs(self.agent1_pos[0] - self.agent2_pos[0]) <= self.vision[0]
            and abs(self.agent1_pos[1] + self.agent2_pos[1] - self.width / 2)
            <= self.vision[1] - 1
        ):
            visible = True

        ag1other = ag2_discrete_pos if visible else 0
        ag2other = ag1_discrete_pos if visible else 0

        # noise1 = random.randint(1, self.width*self.height)
        # noise2 = random.randint(1, self.width*self.height)

        return {
            "agent1": {
                "nosepoke": nosepoke_pos1,
                "water": water_pos1,
                "self": ag1_discrete_pos,
                "otheragent": ag1other,
                "otherpoke": nosepoke_pos2,
            },
            "agent2": {
                "nosepoke": nosepoke_pos2,
                "water": water_pos2,
                "self": ag2_discrete_pos,
                "otheragent": ag2other,
                "otherpoke": nosepoke_pos1,
            },
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # nose poke
        if coords == self.poke_coords1 or coords == self.poke_coords2:
            return {"poke"}

        # drink water
        elif coords == self.water_coords1 or coords == self.water_coords2:
            return {"drink"}

        # no action
        elif action == 4:
            return {"no_action"}

        # check walls.
        if is_agent1:
            bound = self.water_coords1
        else:
            bound = self.water_coords2

        if coords[0] < 0:
            coords[0] = 0
        # elif coords[0] >= self.height:
        #     coords[0] = self.height - 1
        elif coords[0] >= bound[0]:
            coords[0] = bound[0]
        if is_agent1:
            if coords[1] < 0:
                coords[1] = 0
            elif coords[1] >= int(self.width / 2):  # midline
                coords[1] = int(self.width / 2) - 1
        else:
            if coords[1] < self.width / 2:
                coords[1] = int(self.width / 2)
            elif coords[1] >= self.width:
                coords[1] = self.width - 1

        # update agent location

        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif self.poke_coords1 == [r, c]:
                    print("*", end="")
                elif self.poke_coords2 == [r, c]:
                    print("*", end="")
                elif self.water_coords1 == [r, c]:
                    print(".", end="")
                elif self.water_coords2 == [r, c]:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(
            f"{'!SyncPoke!' if self.sync_poke == 1 else '!Miss!' if self.miss == 1 else ''}"
        )
        print(
            f"{'!Agent1Poke!' if ('poke' in self.events['agent1']) else '!Agent1Drink!' if 'drink' in self.events['agent1'] else ''}"
        )
        print(
            f"{'!Agent2Poke!' if 'poke' in self.events['agent2'] else '!Agent2Drink!' if 'drink' in self.events['agent2'] else ''}"
        )

        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()

        ######################################


###### single task where only one side is learning, the other side is trained for random movement
class MultiAgentSing_oneside(MultiAgentEnv):
    def __init__(self, config=None):
        """Config takes in width, height, and ts"""
        config = config or {}
        # Dimensions of the grid.
        self.width = config.get("width", 8)
        self.height = config.get("height", 6)
        self.vision = config.get("Vision", [1, 2])
        self.poke_coords1 = config.get("Poke1", [0, 2])
        self.poke_coords2 = config.get("Poke2", [0, 6])
        self.water_coords1 = config.get("Water1", [5, 2])
        self.water_coords2 = config.get("Water2", [5, 6])
        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 200)
        self.sync_limit = config.get(
            "sync_limit", 2
        )  # default 2 steps but record up to 5
        self.observation_space = Dict(
            {
                "nosepoke": Discrete(self.width * self.height / 2),
                "water": Discrete(self.width * self.height / 2),
                "otheragent": Discrete(self.width * self.height / 2),
                # 0 is unknown location, if known, block 1 to the last
                "self": Discrete(self.width * self.height / 2),
                "otherpoke": Discrete(self.width * self.height / 2)
                # "noise" : Discrete(self.width*self.height),
            }
        )
        # 0=up, 1=right, 2=down, 3=left, 4 = nothing
        self.action_space = Discrete(5)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        self.agent1_pos = [
            random.randint(0, self.height - 1),
            random.randint(0, int(self.width / 2) - 1),
        ]
        self.agent2_pos = [
            random.randint(0, self.height - 1),
            random.randint(int(self.width / 2), self.width - 1),
        ]
        # self.agent1_pos = [0,2]
        # self.agent2_pos = [0,5]
        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # reward availability now
        self.water_available1 = False
        self.water_available2 = False
        self.sync_poke = 0
        self.miss = 0
        self.poke_history1 = np.array([False, False])  # poke in the last time point
        self.poke_history2 = np.array([False, False])
        self.freeze1 = 0
        self.freeze2 = 0
        self.timeout = False
        # Reset agent1's visited fields.
        # self.agent1_visited_fields = set([tuple(self.agent1_pos)])

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """
        self.sync_poke = 0
        self.miss = 0
        # print(self.water_available1)
        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # time of drinking is random
        if self.freeze1 > 0:
            self.freeze1 = self.freeze1 - 1.0  # no action
            r1 = 0
            events1 = []
        else:
            events1 = self._move(self.agent1_pos, action["agent1"], is_agent1=1)

            if self.timeout:
                events1 = []
            if self.water_available1 and "drink" in events1:
                r1 = 2.0
                self.freeze1 = np.random.randint(4)
                self.water_available1 = False
                self.water_coords1 = [
                    random.randint(int(self.height / 2), self.height - 1),
                    random.randint(0, int(self.width / 2) - 1),
                ]  # upper left corner
                self.poke_coords1 = [
                    random.randint(0, int(self.height / 2) - 2),
                    random.randint(0, int(self.width / 2) - 1),
                ]  # upper left corner
            elif (not self.water_available1) and "poke" in events1:
                r1 = 2.0
                self.water_available1 = True
            elif "no_action" in events1:
                r1 = -0.1
            else:
                r1 = -0.1

        if "no_action" in events2:
            r2 = 0
        else:
            r2 = 0.1

        poke_mat = np.array(
            [
                np.append(self.poke_history1[-self.sync_limit :], "poke" in events1),
                np.append(self.poke_history2[-self.sync_limit :], "poke" in events2),
            ]
        )
        # print(poke_mat)
        if np.all(np.any(poke_mat, axis=1)):  # sync poke
            self.sync_poke = 1
            # reset poking history
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
        # if not in sync, give pushnishment to both and create time out - any poking in the next time step does not count
        elif np.any(poke_mat[:, 0]):
            self.poke_history1 = np.repeat(False, 5)
            self.poke_history2 = np.repeat(False, 5)
            self.miss = 1
        else:
            self.poke_history1 = np.append(self.poke_history1[1:], "poke" in events1)
            self.poke_history2 = np.append(self.poke_history2[1:], "poke" in events2)

        # Get observations (based on new agent positions).
        obs = self._get_obs()

        self.agent1_R += r1
        self.agent2_R += r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }
        self.events = {
            "agent1": events1,
            "agent2": events2,
        }

        return obs, rewards, dones, {}  # <- info dict (not needed here).

    def _get_obs(self):
        """
        Returns obs space for one agent using each
        agent's current x/y-positions.
        """

        # discrete coordinate of the locations of nose poke water port and
        nosepoke_pos1 = (
            int(
                self.poke_coords1[0] * self.width / 2
                + (self.poke_coords1[1] % (self.width / 2) + 1)
            )
            - 1
        )
        nosepoke_pos2 = (
            int(
                self.poke_coords2[0] * self.width / 2
                + (self.poke_coords2[1] % (self.width / 2) + 1)
            )
            - 1
        )
        water_pos1 = (
            int(
                self.water_coords1[0] * self.width / 2
                + (self.water_coords1[1] % (self.width / 2) + 1)
            )
            - 1
        )
        water_pos2 = (
            int(
                self.water_coords2[0] * self.width / 2
                + (self.water_coords2[1] % (self.width / 2) + 1)
            )
            - 1
        )
        ag1_discrete_pos = (
            int(
                self.agent1_pos[0] * self.width / 2
                + (self.agent1_pos[1] % (self.width / 2) + 1)
            )
            - 1
        )
        ag2_discrete_pos = (
            int(
                self.agent2_pos[0] * self.width / 2
                + (self.agent2_pos[1] % (self.width / 2) + 1)
            )
            - 1
        )

        ag2other = ag1_discrete_pos
        ag1other = ag2_discrete_pos

        # noise1 = random.randint(1, self.width*self.height)
        # noise2 = random.randint(1, self.width*self.height)

        return {
            "agent1": {
                "nosepoke": nosepoke_pos1,
                "water": water_pos1,
                "self": ag1_discrete_pos,
                "otheragent": ag1other,
                "otherpoke": nosepoke_pos2,
            },
            "agent2": {
                "nosepoke": nosepoke_pos2,
                "water": water_pos2,
                "self": ag2_discrete_pos,
                "otheragent": ag2other,
                "otherpoke": nosepoke_pos1,
            },
        }

    def _move(self, coords, action, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # nose poke
        if coords == self.poke_coords1 or coords == self.poke_coords2:
            return {"poke"}

        # drink water
        elif coords == self.water_coords1 or coords == self.water_coords2:
            return {"drink"}

        # no action
        elif action == 4:
            return {"no_action"}

        # check walls.
        if is_agent1:
            bound = self.water_coords1
        else:
            bound = self.water_coords2

        if coords[0] < 0:
            coords[0] = 0
        # elif coords[0] >= self.height:
        #     coords[0] = self.height - 1
        elif coords[0] >= bound[0]:
            coords[0] = bound[0]
        if is_agent1:
            if coords[1] < 0:
                coords[1] = 0
            elif coords[1] >= int(self.width / 2):  # midline
                coords[1] = int(self.width / 2) - 1
        else:
            if coords[1] < self.width / 2:
                coords[1] = int(self.width / 2)
            elif coords[1] >= self.width:
                coords[1] = self.width - 1

        # update agent location

        # No new tile for agent1.
        return set()

    def render(self, mode=None):
        print("_" * (self.width + 2))
        for r in range(self.height):
            print("|", end="")
            for c in range(self.width):
                field = r * self.width + c % self.width
                if self.agent1_pos == [r, self.width / 2 - 1 - c]:
                    print("1", end="")
                elif self.agent2_pos == [r, c]:
                    print("2", end="")
                elif self.poke_coords1 == [r, self.width / 2 - 1 - c]:
                    print("*", end="")
                elif self.poke_coords2 == [r, c]:
                    print("*", end="")
                elif self.water_coords1 == [r, self.width / 2 - 1 - c]:
                    print(".", end="")
                elif self.water_coords2 == [r, c]:
                    print(".", end="")
                else:
                    print(" ", end="")
            print("|")
        print("‾" * (self.width + 2))
        print(
            f"{'!SyncPoke!' if self.sync_poke == 1 else '!Miss!' if self.miss == 1 else ''}"
        )
        print(
            f"{'!Agent1Poke!' if ('poke' in self.events['agent1']) else '!Agent1Drink!' if 'drink' in self.events['agent1'] else ''}"
        )
        print(
            f"{'!Agent2Poke!' if 'poke' in self.events['agent2'] else '!Agent2Drink!' if 'drink' in self.events['agent2'] else ''}"
        )

        print("R1={: .1f}".format(self.agent1_R))
        print("R2={: .1f}".format(self.agent2_R))
        print()
