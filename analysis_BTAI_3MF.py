from agent.inference.TemporalSliceBuilder import TemporalSliceBuilder
from env.dSpritesEnv import dSpritesEnv
from env.wrapper.dSpritesPreProcessingWrapper import dSpritesPreProcessingWrapper
from agent.BTAI_3MF import BTAI_3MF
from analysis.GUI import GUI


def main():
    """
    Display a graphical user interface to analyse the behaviour of BTAI_3MF.
    :return: nothing.
    """

    # Create the environment.
    env = dSpritesEnv(granularity=1, repeat=1)
    env = dSpritesPreProcessingWrapper(env)

    # Define the parameters of the generative model.
    a = env.a()
    b = env.b()
    c = env.c()
    d = env.d(uniform=True)

    # Define the temporal slice structure.
    ts = TemporalSliceBuilder("A_1", env.n_actions) \
        .add_state("S_shape", d["S_shape"]) \
        .add_state("S_scale", d["S_scale"]) \
        .add_state("S_orientation", d["S_orientation"]) \
        .add_state("S_pos_x", d["S_pos_x"]) \
        .add_state("S_pos_y", d["S_pos_y"]) \
        .add_observation("O_shape", a["O_shape"], ["S_shape"]) \
        .add_observation("O_scale", a["O_scale"], ["S_scale"]) \
        .add_observation("O_orientation", a["O_orientation"], ["S_orientation"]) \
        .add_observation("O_pos_x", a["O_pos_x"], ["S_pos_x"]) \
        .add_observation("O_pos_y", a["O_pos_y"], ["S_pos_y"]) \
        .add_transition("S_shape", b["S_shape"], ["S_shape"]) \
        .add_transition("S_scale", b["S_scale"], ["S_scale"]) \
        .add_transition("S_orientation", b["S_orientation"], ["S_orientation"]) \
        .add_transition("S_pos_x", b["S_pos_x"], ["S_pos_x", "A_1"]) \
        .add_transition("S_pos_y", b["S_pos_y"], ["S_pos_y", "A_1"]) \
        .add_preference(["O_pos_x", "O_pos_y", "O_shape"], c["O_shape_pos_x_y"]) \
        .build()

    # Create the agent.
    agent = BTAI_3MF(ts, max_planning_steps=100, exp_const=2.4)

    # Create the GUI for analysis.
    gui = GUI(env, agent)
    gui.loop()


if __name__ == '__main__':
    main()
