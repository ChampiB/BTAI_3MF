import time

from agent.inference.TemporalSliceBuilder import TemporalSliceBuilder
from env.dSpritesEnv import dSpritesEnv
from env.wrapper.dSpritesPreProcessingWrapper import dSpritesPreProcessingWrapper
from agent.BTAI_3MF import BTAI_3MF
import torch

# ------------------------------------------------------------------------------ #
# Results:                                                                       #
# ------------------------------------------------------------------------------ #
# dSpritesEnv(granularity=8, repeat=8) + max_planning_steps=50 + exp_const=2.4)  #
# Percentage of task solved: 0.895625                                            #
# Execution time(sec): 1.2799999713897705 + / - 12.800000190734863               #
# ------------------------------------------------------------------------------ #
# dSpritesEnv(granularity=4, repeat=8) + max_planning_steps=50 + exp_const=2.4)  #
# Percentage of task solved: 0.9778125                                           #
# Execution time (sec):  1.2799999713897705 +/- 12.800000190734863               #
# ------------------------------------------------------------------------------ #
# dSpritesEnv(granularity=2, repeat=8) + max_planning_steps=50 + exp_const=2.4)  #
# Percentage of task solved: 0.9965625                                           #
# Execution time(sec): 1.2799999713897705 + / - 12.800000190734863               #
# ------------------------------------------------------------------------------ #
# dSpritesEnv(granularity=1, repeat=8) + max_planning_steps=50 + exp_const=2.4)  #
# Percentage of task solved: 0.72                                                #
# Execution time(sec): 2.559999942779541 + / - 18.010276794433594                #
#                                                                                #
# dSpritesEnv(granularity=1, repeat=8) + max_planning_steps=100 + exp_const=2.4) #
# Percentage of task solved: 0.77                                                #
# Execution time(sec): 5.119999885559082 + / - 25.209136962890625                #
#                                                                                #
# dSpritesEnv(granularity=1, repeat=8) + max_planning_steps=150 + exp_const=2.4) #
# Percentage of task solved: 1.0                                                 #
# Execution time (sec): 2.559999942779541 +/- 18.010276794433594                 #
# ------------------------------------------------------------------------------ #


def main():
    """
    A simple example of how to use the BTAI_3MF framework.
    :return: nothing.
    """

    # Create the environment.
    env = dSpritesEnv(granularity=4, repeat=8)
    env = dSpritesPreProcessingWrapper(env)

    # Define the parameters of the generative model.
    a = env.a()
    b = env.b()
    c = env.c()
    d = env.d(uniform=True)

    # Define the temporal slice structure.
    ts = TemporalSliceBuilder("A_0", env.n_actions) \
        .add_state("S_pos_x", d["S_pos_x"]) \
        .add_state("S_pos_y", d["S_pos_y"]) \
        .add_state("S_shape", d["S_shape"]) \
        .add_state("S_scale", d["S_scale"]) \
        .add_state("S_orientation", d["S_orientation"]) \
        .add_observation("O_pos_x", a["O_pos_x"], ["S_pos_x"]) \
        .add_observation("O_pos_y", a["O_pos_y"], ["S_pos_y"]) \
        .add_observation("O_shape", a["O_shape"], ["S_shape"]) \
        .add_observation("O_scale", a["O_scale"], ["S_scale"]) \
        .add_observation("O_orientation", a["O_orientation"], ["S_orientation"]) \
        .add_transition("S_pos_x", b["S_pos_x"], ["S_pos_x", "A_0"]) \
        .add_transition("S_pos_y", b["S_pos_y"], ["S_pos_y", "A_0"]) \
        .add_transition("S_shape", b["S_shape"], ["S_shape"]) \
        .add_transition("S_scale", b["S_scale"], ["S_scale"]) \
        .add_transition("S_orientation", b["S_orientation"], ["S_orientation"]) \
        .add_preference(["O_pos_x", "O_pos_y", "O_shape"], c["O_shape_pos_x_y"]) \
        .build()

    # Create the agent.
    agent = BTAI_3MF(ts, max_planning_steps=50, exp_const=2.4)

    # Implement the action-perception cycles.
    n_trials = 100
    score = 0
    ex_times_s = torch.zeros([n_trials])
    for i in range(n_trials):
        obs = env.reset()
        env.render()
        agent.reset(obs)
        ex_times_s[i] = time.time()
        while not env.done():
            action = agent.step()
            obs = env.execute(action)
            env.render()
            agent.update(action, obs)
        ex_times_s[i] = time.time() - ex_times_s[i]
        score += env.get_reward()

    # Display the performance of the agent.
    print("Percentage of task solved: {}".format((score + n_trials) / (2 * n_trials)))
    print("Execution time (sec): {} +/- {}".format(ex_times_s.mean().item(), ex_times_s.std(dim=0).item()))


if __name__ == '__main__':
    main()
