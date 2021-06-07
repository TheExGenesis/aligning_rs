from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.agents.pg.atari import AtariFfAgent
import torch
import imageio
import numpy as np
from matplotlib.pyplot import imshow


def makeAtariGif(env, agent, length=58, fps=29, path='./atari.gif'):
    print("Making gif")
    images = []
    done = False
    obs, _, _, info = env.step(-1)
    obs = torch.from_numpy(obs).unsqueeze(0)
    prev_action = torch.tensor(0.0, dtype=torch.float)  # None
    prev_reward = torch.tensor(0.0, dtype=torch.float)  # None
    for i in range(length):
        if done:
            break
        step = agent.step(obs, prev_action, prev_reward)
        obs, rewards, done, info = env.step(step.action)
        img = env.get_obs()[-1]
        images.append(img)
        # Next stat
        obs = torch.from_numpy(obs).unsqueeze(0)
        prev_action = step.action.clone()
        prev_reward = torch.tensor([rewards], dtype=torch.float)
    imageio.mimsave(path, [np.array(img)
                           for i, img in enumerate(images) if i % 2 == 0], fps=15)
    return

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--game', help='Atari game', default='pong')
#     parser.add_argument(
#         '--run_ID', help='run identifier (logging)', type=int, default=0)
#     parser.add_argument('--cuda_idx', help='gpu to use ',
#                         type=int, default=None)
#     parser.add_argument('--sample_mode', help='serial or parallel sampling',
#                         type=str, default='serial', choices=['serial', 'cpu', 'gpu', 'alternating'])
#     parser.add_argument(
#         '--n_parallel', help='number of sampler workers', type=int, default=2)
#     args = parser.parse_args()
#     build_and_train(
#         game=args.game,
#         run_ID=args.run_ID,
#         cuda_idx=args.cuda_idx,
#         sample_mode=args.sample_mode,
#         n_parallel=args.n_parallel,
#     )
