# run one iteration and get observation vars
import numpy as np


def run_simulation(env, model):
    tot_reward = np.zeros(2)
    t_close = 0  # time steps where the two agents are within 1 step
    t_mid = np.zeros(2)  # time steps where the agents are in the middle lane
    actions1 = list()
    actions2 = list()
    pos1x = list()
    pos2x = list()
    pos1y = list()
    pos2y = list()
    miss = list()
    correct = list()
    poke1 = list()
    poke2 = list()
    drink1 = list()
    drink2 = list()
    t = -1

    obs = env.reset()
    state1 = [np.zeros(256, np.float32) for _ in range(2)]
    state2 = [np.zeros(256, np.float32) for _ in range(2)]
    while True:
        t += 1
        a1, state1, _ = model.compute_single_action(
            observation=obs["agent1"], state=state1, policy_id="policy1"
        )
        a2, state2, _ = model.compute_single_action(
            observation=obs["agent2"], state=state2, policy_id="policy2"
        )
        obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})
        actions1.append(a1)
        actions2.append(a2)
        pos1y.append(env.agent1_pos[0])
        pos2y.append(env.agent2_pos[0])
        pos1x.append(env.agent1_pos[1])
        pos2x.append(env.agent2_pos[1])
        tot_reward[0] = tot_reward[0] + rewards["agent1"]
        tot_reward[1] = tot_reward[1] + rewards["agent2"]
        if env.miss == 1:
            miss.append(t)
        if env.sync_poke == 1:
            correct.append(t)
        if "poke" in env.events["agent1"]:
            poke1.append(t)
        if "poke" in env.events["agent2"]:
            poke2.append(t)
        if "drink" in env.events["agent1"]:
            drink1.append(t)
        if "drink" in env.events["agent2"]:
            drink2.append(t)
        if dones["agent1"]:
            break

    return (
        {
            "ncorrect": env.ncorrect,
            "nmiss1": env.nmiss1,
            "nmiss2": env.nmiss2,
            "ncorrect1": env.ncorrect1,
            "ncorrect2": env.ncorrect2,
            "npoke1": env.npoke1,
            "npoke2": env.npoke2,
            "ndrink1": env.ndrink1,
            "ndrink2": env.ndrink2,
        },
        {"correct": correct, "miss": miss},
        {"poke1": poke1, "poke2": poke2, "drink1": drink1, "drink2": drink2},
        {"ag1": tot_reward[0], "ag2": tot_reward[1]},
        {"ag1": actions1, "ag2": actions2},
        {"ag1x": pos1x, "ag2x": pos2x, "ag1y": pos1y, "ag2y": pos2y},
    )


def run_simulation_2ag(env, model1, model2, model1pol, model2pol):
    tot_reward = np.zeros(2)
    t_close = 0  # time steps where the two agents are within 1 step
    t_mid = np.zeros(2)  # time steps where the agents are in the middle lane
    actions1 = list()
    actions2 = list()
    pos1x = list()
    pos2x = list()
    pos1y = list()
    pos2y = list()
    miss = list()
    correct = list()
    poke1 = list()
    poke2 = list()
    t = 0

    obs = env.reset()
    state1 = [np.zeros(256, np.float32) for _ in range(2)]
    state2 = [np.zeros(256, np.float32) for _ in range(2)]
    while True:
        t += 1
        a1, state1, _ = model1.compute_single_action(
            observation=obs["agent1"], state=state1, policy_id=model1pol
        )
        a2, state2, _ = model2.compute_single_action(
            observation=obs["agent2"], state=state2, policy_id=model2pol
        )
        obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})
        actions1.append(a1)
        actions2.append(a2)
        pos1y.append(env.agent1_pos[0])
        pos2y.append(env.agent2_pos[0])
        pos1x.append(env.agent1_pos[1])
        pos2x.append(env.agent2_pos[1])
        tot_reward[0] = tot_reward[0] + rewards["agent1"]
        tot_reward[1] = tot_reward[1] + rewards["agent2"]
        # # record the number of steps the two agents stayed in the middle lane
        # if env.agent1_pos[1] == env.width/2-1:
        #     t_mid[0] += 1
        # if env.agent2_pos[1] == env.width/2:
        #     t_mid[1] += 1
        # if abs(env.agent1_pos[0] - env.agent2_pos[0]) <= 1:
        #     t_close += 1
        if env.miss == 1:
            miss.append(t)
        if env.sync_poke == 1:
            correct.append(t)
        if "poke" in env.events["agent1"]:
            poke1.append(t)
        if "poke" in env.events["agent2"]:
            poke2.append(t)
        if dones["agent1"]:
            break

    return (
        {"ncorrect": env.ncorrect, "nmiss1": env.nmiss1, "nmiss2": env.nmiss2},
        {"correct": correct, "miss": miss},
        {"ag1": poke1, "ag2": poke2},
        {"ag1": tot_reward[0], "ag2": tot_reward[1]},
        {"ag1": actions1, "ag2": actions2},
        {"ag1x": pos1x, "ag2x": pos2x, "ag1y": pos1y, "ag2y": pos2y},
    )


def run_simulation_sep(env, model):
    tot_reward = np.zeros(2)
    t_close = np.zeros(1)
    t_mid = np.zeros(2)
    n_sync = np.zeros(1)
    n_poke = np.zeros(2)
    obs = env.reset()
    while True:
        a1 = model.compute_single_action(obs["agent1"], policy_id="policy1")
        a2 = model.compute_single_action(obs["agent2"], policy_id="policy2")
        obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})
        tot_reward[0] = tot_reward[0] + rewards["agent1"]
        tot_reward[1] = tot_reward[1] + rewards["agent2"]
        # record the number of steps the two agents stayed in the middle lane
        if env.agent1_pos[1] == env.width / 2 - 1:
            t_mid[0] = t_mid[0] + 1
        if env.agent2_pos[1] == env.width / 2:
            t_mid[1] = t_mid[1] + 1
        # record the number of steps when the two agents stays within one step away from each other
        if abs(env.agent1_pos[0] - env.agent2_pos[0]) <= 1:
            t_close = t_close + 1
        if env.sync_poke == 1:
            n_sync = n_sync + 1
        if "poke" in env.events["agent1"]:
            n_poke[0] = n_poke[0] + 1
        if "poke" in env.events["agent2"]:
            n_poke[1] = n_poke[1] + 1
        if dones["agent1"]:
            break
    return t_close, t_mid, n_sync, n_poke, tot_reward


def run_simulation_call(env, model):
    tot_reward = np.zeros(2)
    t_close = np.zeros(1)
    t_mid = np.zeros(2)
    n_sync = np.zeros(1)
    n_miss = np.zeros(1)
    n_poke = np.zeros(2)
    n_call = np.zeros(2)
    obs = env.reset()
    while True:
        a1 = model.compute_single_action(obs["agent1"], policy_id="policy1")
        a2 = model.compute_single_action(obs["agent2"], policy_id="policy2")
        obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})
        tot_reward[0] = tot_reward[0] + rewards["agent1"]
        tot_reward[1] = tot_reward[1] + rewards["agent2"]
        # record the number of steps the two agents stayed in the middle lane
        if env.agent1_pos[1] == env.width / 2 - 1:
            t_mid[0] = t_mid[0] + 1
        if env.agent2_pos[1] == env.width / 2:
            t_mid[1] = t_mid[1] + 1
        # record the number of steps when the two agents stays within one step away from each other
        if abs(env.agent1_pos[0] - env.agent2_pos[0]) <= 1:
            t_close = t_close + 1
        if env.miss == 1:
            n_miss = n_miss + 1
        if env.sync_poke == 1:
            n_sync = n_sync + 1
        if "poke" in env.events["agent1"]:
            n_poke[0] = n_poke[0] + 1
        if "poke" in env.events["agent2"]:
            n_poke[1] = n_poke[1] + 1
        # calls
        if a1 == 5:
            n_call[0] = n_call[0] + 1
        if a2 == 5:
            n_call[1] = n_call[1] + 1
        if dones["agent1"]:
            break
    return t_close, t_mid, n_sync, n_miss, n_poke, n_call, tot_reward
