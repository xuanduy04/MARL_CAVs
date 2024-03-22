"""
This environment is built on HighwayEnv with one main road and one merging lane.
Dong Chen: chendon9@msu.edu
Date: 01/05/2021
"""
import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle


class MergeMultilanePriorityEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    n_obs_features = 6
    n_a = 5
    n_s = n_obs_features * 6

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "see_priority_vehicle": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True
            },
            "controlled_vehicles": 1,
            "screen_width": 608,
            "screen_height": 128,
            "centering_position": [0.3, 0.5],
            "scaling": 3,
            "simulation_frequency": 15,  # [Hz]
            "duration": 20,  # time step
            "policy_frequency": 5,  # [Hz]
            "reward_speed_range": [10, 30],
            "priority_target_speed_range": [30, 40],
            "COLLISION_REWARD": 200,  # default=200
            "HIGH_SPEED_REWARD": 1,  # default=1
            "PRIORITY_SPEED_COST": 1,
            "HEADWAY_COST": 4,  # default=4
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 4,  # default=4
            "PRIORITY_LANE_COST": 1,
            "LANE_CHANGE_COST": 0.5,  # default=0.5
            "num_CAV": 4,
            "num_HDV": 4,
            "flatten_obs": False, # set this to False for attention to work
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action[idx], vehicle) for idx, vehicle in enumerate(self.controlled_vehicles)) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 2):
            Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
                    10 * self.ends[2]))
        else:
            Merging_lane_cost = 0

        # lane change cost to avoid unnecessary/frequent lane changes
        lane_change_cost = -1 * self.config["LANE_CHANGE_COST"] if action == 0 or action == 2 else 0

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        
        # -- Priority vehicle related --
        # idea: do not punish vehicles who don't need to dodge.
        #       while punishing vehicles who do

        # compute cost for blocking the priority vehicle's path
        priority_vehicle_dist, priority_vehicle = self.road.priority_vehicle_relative_position(vehicle)
        priority_lane_cost = -1 * self.config["PRIORITY_LANE_COST"] \
            if priority_vehicle_dist < 0 and vehicle.lane_index == priority_vehicle.lane_index else 0
        
        # if you are in the process of dodging, I'll reduce the blocking cost.
        if priority_lane_cost and (action == 0 or action == 2):
            priority_lane_cost *= 0.5
            # and nullify the lane change cost
            lane_change_cost = 0
        
        # cost for slowing the priority vehicle.
        # priority_scaled_speed = \
        #     utils.lmap(priority_vehicle.speed, self.config["priority_target_speed_range"], [0, 1]) \
        #         if priority_vehicle_dist < 0 else 0
        # Note, reward clipped to avoid rewarding vehicles that doesn't (need to) dodge the priority vehicle.

        # Add the below line iff you want to use the above cost.  
        # + (self.config["PRIORITY_SPEED_COST"] * priority_scaled_speed if priority_scaled_speed < 0 else 0) \
        
        # compute overall reward
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + (self.config["HEADWAY_COST"] * Headway_cost if Headway_cost < 0 else 0) \
                 + priority_lane_cost \
                 + lane_change_cost
        return reward

    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # vehicle is on the main road
            if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("a", "b", 1) or vehicle.lane_index == (
                    "b", "c", 0) or vehicle.lane_index == ("b", "c", 1) or vehicle.lane_index == (
                    "c", "d", 0) or vehicle.lane_index == ("c", "d", 1):
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe vehicles on the ramp
                elif vehicle.lane_index == ("a", "b", 1) and vehicle.position[0] > self.ends[0]:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))
                else:
                    v_fr, v_rr = None, None
            else:
                # vehicle is on the ramp
                v_fr, v_rr = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 1))
                else:
                    v_fl, v_rl = None, None

            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if type(v) is MDPVehicle and v is not None:
                    neighbor_vehicle.append(v)
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        obs, reward, done, info = super().step(action)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info

        for vehicle in self.controlled_vehicles:
            vehicle.local_reward = self._agent_reward(action, vehicle)
        # local reward
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # regional reward
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        if self.config["flatten_obs"]:
            obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self, is_training=False) -> None:
        self._make_road()

        num_CAV = self.config["num_CAV"]
        num_HDV = self.config["num_HDV"]
        if is_training:
            # chance to train with less vehicles.
            # Simulates curriculum training, in a way.
            num_CAV = np.random.choice(np.arange(max(1,num_CAV-2), num_CAV+1), 1)[0]
            num_HDV = np.random.choice(np.arange(max(1,num_HDV-2), num_HDV+1), 1)[0]

        self._make_vehicles(num_CAV=num_CAV, num_HDV=num_HDV)
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(self.ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c",
                         StraightLane([sum(self.ends[:2]), y[i]], [sum(self.ends[:3]), y[i]],
                                      line_types=line_type_merge[i]))
            net.add_lane("c", "d",
                         StraightLane([sum(self.ends[:3]), y[i]], [sum(self.ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [self.ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(self.ends[0], -amplitude), ljk.position(sum(self.ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * self.ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(self.ends[1], 0), lkb.position(self.ends[1], 0) + [self.ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))
        self.road = road

    def _make_vehicles(self, num_CAV, num_HDV) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        priority_vehicles_type = utils.class_from_path(self.config["priority_vehicles_type"])
        self.controlled_vehicles = []
        road.priority_vehicle = None

        num_PV = 1

        spawn_points_s1 = [10, 50, 90, 130] #, 170, 210, 250]
        spawn_points_s2 = [5, 45, 85, 125] #, 165, 205, 245]
        spawn_points_m = [5, 45, 85, 125] #, 165, 205]

        """Spawn points for PV"""
        # for now, PV always spawn on straight road.
        # if not self.config["priority_vehicle_can_spawn_first"]:
            # print("priority_vehicle_can_spawn_first setting is currently unimplemented")

        # priority_vehicle's spawn lane & spawn point index
        spawn_lane_pv = np.random.choice(np.arange(0, 2), 1)[0]
        spawn_point_pv = None

        # TODO: optimize code?
        if spawn_lane_pv == 0:
            spawn_lane_pv = ("a", "b", 0)
            spawn_point_pv = np.random.choice(spawn_points_s1, num_PV, replace=False)[0]
            spawn_points_s1.remove(spawn_point_pv)
        elif spawn_lane_pv == 1:
            spawn_lane_pv = ("a", "b", 1)
            spawn_point_pv = np.random.choice(spawn_points_s2, num_PV, replace=False)[0]
            spawn_points_s2.remove(spawn_point_pv)
        else:
            raise NotImplementedError
            # spawn_lane_pv = ("j", "k", 0)
        
        """Spawn points for CAV"""
        # spawn point indexes on the straight road
        spawn_point_s_c1 = np.random.choice(spawn_points_s1, num_CAV // 3, replace=False)
        spawn_point_s_c2 = np.random.choice(spawn_points_s2, num_CAV // 3, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - 2 * num_CAV // 3,
                                           replace=False)
        spawn_point_s_c1 = list(spawn_point_s_c1)
        spawn_point_s_c2 = list(spawn_point_s_c2)
        spawn_point_m_c = list(spawn_point_m_c)

        # remove the points to avoid duplicate
        for a in spawn_point_s_c1:
            spawn_points_s1.remove(a)
        for b in spawn_point_s_c2:
            spawn_points_s2.remove(b)
        for c in spawn_point_m_c:
            spawn_points_m.remove(c)

        """Spawn points for HDV"""
        # spawn point indexes on the straight road
        spawn_point_s_h1 = np.random.choice(spawn_points_s1, num_HDV // 3, replace=False)
        spawn_point_s_h2 = np.random.choice(spawn_points_s2, num_HDV // 3, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(spawn_points_m, num_HDV - 2 * num_HDV // 3,
                                           replace=False)
        spawn_point_s_h1 = list(spawn_point_s_h1)
        spawn_point_s_h2 = list(spawn_point_s_h2)
        spawn_point_m_h = list(spawn_point_m_h)

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_CAV + num_HDV + num_PV) * 2 + 25  # range from [25, 27]
        loc_noise = np.random.rand(num_CAV + num_HDV + num_PV) * 3 - 1.5  # range from [-1.5, 1.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        """spawn the PV"""
        road.priority_vehicle = \
            priority_vehicles_type(road, road.network.get_lane(spawn_lane_pv).position(
                spawn_point_pv + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
        road.vehicles.append(road.priority_vehicle)

        """spawn the CAV on the straight road first"""
        for _ in range(num_CAV // 3):
            ego_vehicle1 = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 0)).position(
                spawn_point_s_c1.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle1)
            road.vehicles.append(ego_vehicle1)

        for _ in range(num_CAV // 3):
            ego_vehicle2 = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 1)).position(
                spawn_point_s_c2.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle2)
            road.vehicles.append(ego_vehicle2)

        """spawn the rest CAV on the merging road"""
        for _ in range(num_CAV - 2 * num_CAV // 3):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV // 3):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
                    spawn_point_s_h1.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0)))

        for _ in range(num_HDV // 3):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(
                    spawn_point_s_h2.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0)))

        """spawn the rest HDV on the merging road"""
        for _ in range(num_HDV - 2 * num_HDV // 3):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


class MergeMultilanePriorityEnvMARL(MergeMultilanePriorityEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 6, 
                    "see_priority_vehicle": True,
                }},
            "controlled_vehicles": 4
        })
        return config


register(
    id='merge-multilane-priority-v0',
    entry_point='highway_env.envs:MergeMultilanePriorityEnv',
)

register(
    id='merge-multilane-priority-multi-agent-v0',
    entry_point='highway_env.envs:MergeMultilanePriorityEnvMARL',
)
