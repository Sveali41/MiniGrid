import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from minigrid_custom_env import *
from minigrid.wrappers import FullyObsWrapper
import hydra
from modelBased.common.utils import TRAINER_PATH
from omegaconf import DictConfig
from modelBased.data_collect import run_env, save_experiments

@hydra.main(version_base=None, config_path = str(TRAINER_PATH / "conf"), config_name="config")
def data_collect(cfg: DictConfig):
    level_path = cfg.generator.level_path
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path=level_path, custom_mission="pick up the yellow ball",
                                        max_steps=2000, render_mode='human'))
    obs, obs_next, act,rew, done = run_env(env, cfg.world_model)
    save_experiments(cfg.world_model,obs,obs_next, act, rew, done)

if __name__ == "__main__": 
    data_collect()