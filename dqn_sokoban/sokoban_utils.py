import matplotlib.pyplot as plt

from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast


def state_to_pic(state):
    dim_room = (state.shape[0], state.shape[1])
    env = SokobanEnvFast(dim_room, 2)
    env.restore_full_state_from_np_array_version(state)
    return env.render(mode='rgb_array').astype(int)


def show_state(state):
    pic = state_to_pic(state)
    plt.clf()
    plt.imshow(pic)
    plt.show()