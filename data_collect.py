from minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper
from path import *
import pandas as pd
import json

# collect the data buffer
path = Paths()
env = ImgObsWrapper(CustomEnvFromFile(txt_file_path=path.LEVEL_FILE, custom_mission="Find the key "
                                                                                    "and open the "
                                                                                    "door.", render_mode="human"))
time_step = 10
obs, _ = env.reset()
data_buffer = dict()
data_buffer['obs'] = list()
data_buffer['obs'].append(obs)
data_buffer['next_obs'] = list()
data_buffer['reward'] = list()
data_buffer['action'] = list()
data_buffer['terminated'] = list()

for i in range(time_step):
    action = env.action_space.sample()
    data_buffer['action'].append(action)
    next_obs, reward, terminated, truncated, info = env.step(action)
    data_buffer['next_obs'].append(next_obs)
    data_buffer['reward'].append(reward)
    data_buffer['terminated'].append(terminated)
    obs = next_obs
    data_buffer['obs'].append(obs)

# save the data to csv
obs_save = os.path.join(path.MODEL_BASED_DATA, 'obs.json')
action_save = os.path.join(path.MODEL_BASED_DATA, 'action.json')
reward_save = os.path.join(path.MODEL_BASED_DATA, 'reward.json')
next_obs_save = os.path.join(path.MODEL_BASED_DATA, 'obs_next.json')

df = pd.DataFrame({'obs': data_buffer['obs'][:-1], 'action': data_buffer['action'], 'reward': data_buffer['reward'],
                   'next_obs': data_buffer['next_obs']})
# Save the DataFrame to a CSV file
data_save = os.path.join(path.MODEL_BASED_DATA, 'env_data.csv')
df.to_csv(data_save, index=False)
