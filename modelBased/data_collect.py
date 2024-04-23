from minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper
from path import *
import pandas as pd
import json

# collect the data buffer
path = Paths()
env = FullyObsWrapper(CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key "
                                                                                      "and open the "
                                                                                      "door.",
                                        max_steps=2000, render_mode="human"))
# time_step
time_step = 10000
obs, _ = env.reset()
data_buffer = dict()
data_buffer['obs'] = list()
data_buffer['obs'].append(obs['image'].tolist())
data_buffer['next_obs'] = list()
data_buffer['reward'] = list()
data_buffer['action'] = list()
data_buffer['terminated'] = list()

for i in range(time_step):
    action = env.action_space.sample()
    data_buffer['action'].append(int(action))
    next_obs, reward, terminated, truncated, info = env.step(action)
    data_buffer['next_obs'].append(next_obs['image'].tolist())
    data_buffer['reward'].append(float(reward))
    data_buffer['terminated'].append(terminated)
    obs = next_obs
    data_buffer['obs'].append(obs['image'].tolist())
    if terminated or truncated:
        env.reset()

# Save the DataFrame to a CSV file
data_buffer['obs'] = data_buffer['obs'][:-1]
data_save = os.path.join(path.MODEL_BASED_DATA, 'env_data.json')
with open(data_save, 'w') as json_file:
    json.dump(data_buffer, json_file, indent=4)
