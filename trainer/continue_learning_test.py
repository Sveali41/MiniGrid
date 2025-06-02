from omegaconf import DictConfig
import Support
from generator.common.utils import load_gen
from modelBased.common.utils import TRAINER_PATH
from modelBased import AttentionWM_training, PPO_world_training
from datetime import datetime
import hydra
import os
import torch


'''
Process
1. load the generator
2. use the generator to generator env 
(comparision among the different env as loss1)
3. collect data from the env
4. train(finetuning) the attention & WM
5. using the trained attention & WM to play in the final task sets
6. return score in the final task as the feedback

'''


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def collect_data(cfg: DictConfig):
    support = Support.Support(cfg)
    support.del_env_data_file()  # clear the data_save_path
    env_text_file_name = ['env2_with_keydoor.txt']
    file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'level'))
    step_len = len(env_text_file_name)

    for step in range(step_len):
        print(f"Step {step+1} of {step_len}...")
        # env = support.wrap_env(support.generate_env(model))
        file_path = os.path.join(file_dir, env_text_file_name[step])
        env = support.wrap_env_from_text(file_path)
        file_name = os.path.splitext(env_text_file_name[step])[0]  # 'env1_move'
        data_save_dir = '/home/siyao/project/rlPractice/MiniGrid/trainer/data'
        cfg.env.collect.data_save_path = os.path.join(data_save_dir, f'{file_name}.npz')
        support.collect_data_trainer(env)


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def test_1(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from fisher_buffer import FisherReplayBuffer
    from modelBased.AttentionWM import AttentionWorldModel
    import numpy as np

    fisher_buffer = FisherReplayBuffer(max_size=10000)
    old_params, fisher = None, None

    env_text_file_name = ['env1_test.txt', 'env2_test.txt', 'env2_move.txt', 'env3_move.txt', 'env4_move.txt']
    # env_text_file_name = ['env1_test.txt']
    step_len = len(env_text_file_name)
    data_save_dir = '/home/siyao/project/rlPractice/MiniGrid/trainer/data'

    for step in range(step_len):
        print(f"Step {step+1} of {step_len}...")

        # === 设置当前任务路径 ===
        file_name = os.path.splitext(env_text_file_name[step])[0]
        cfg.attention_model.data_dir = os.path.join(data_save_dir, f'{file_name}.npz')

        # === 混合 replay ===
        if len(fisher_buffer) > 0:
            replay_data = fisher_buffer.export_dict()
        else:
            replay_data = None

        # === 启动训练（含 EWC） ===
        cur_old_params, cur_fisher = AttentionWM_training.train_api(cfg, old_params, fisher, replay_data=replay_data)
        old_params, fisher = cur_old_params, cur_fisher
        # === 更新 Fisher Replay Buffer ===
        task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
        samples = {
            'obs': task_npz['a'],
            'obs_next': task_npz['b'],
            'act': task_npz['c'],
        }
        model_eval = AttentionWorldModel(cfg.attention_model).to(device)
        # fisher_buffer.update_with_top_k_recent(samples, model=model_eval, fisher=fisher, recent_k=20000, top_k=10000)
        fisher_buffer.update_with_random_by_ratio(samples, 0.4)
        # *add the function: add data to the fisher buffer by proprotional sampling

    # === 最后评估 ===
    cfg.attention_model.freeze_weight = True
    for test_file in ['env1_test.npz', 'env2_test.npz',  'env2_move.npz', 'env3_move.npz']:
    # for test_file in ['env1_test.npz']:
        cfg.attention_model.data_dir = os.path.join(data_save_dir, test_file)
        AttentionWM_training.train_api(cfg)



@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def test_2(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_params, fisher = None, None
    env_text_file_name = ['env1_test.txt']
    step_len = len(env_text_file_name)

    for step in range(step_len):
        print(f"Step {step+1} of {step_len}...")
        # env = support.wrap_env(support.generate_env(model))
        file_name = os.path.splitext(env_text_file_name[step])[0]  # 'env1_move'
        data_save_dir = '/home/siyao/project/rlPractice/MiniGrid/trainer/data'
        cfg.attention_model.data_dir = os.path.join(data_save_dir, f'{file_name}.npz')
        cur_old_params, cur_fisher = AttentionWM_training.train_api(cfg, old_params, fisher)
        old_params, fisher = cur_old_params, cur_fisher

    cfg.attention_model.freeze_weight = True
    cfg.attention_model.data_dir = '/home/siyao/project/rlPractice/MiniGrid/trainer/data/env1_test.npz'
    AttentionWM_training.train_api(cfg, old_params, fisher)
    cfg.attention_model.data_dir = '/home/siyao/project/rlPractice/MiniGrid/trainer/data/env2_test.npz'
    AttentionWM_training.train_api(cfg, old_params, fisher)

    
if __name__ == "__main__":
    # collect_data()
    test_1()
    # test_2()