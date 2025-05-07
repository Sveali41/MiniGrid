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


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config")
def run(cfg: DictConfig):
    old_params, fisher = None, None

    support = Support.Support(cfg)

    # generate final task
    final_task_set = support.generate_final_task(18, 18, 5)

    # train the policy in the final task set
    base_model_dir = cfg.PPO.real_policy_folder
    policy_file_names = []
    for map_path in final_task_set:
        cfg.PPO.env_path = map_path
        map_filename = os.path.splitext(os.path.basename(map_path))[0]
        policy_file_name = f"PPO_{map_filename}.ckpt"
        cfg.PPO.real_policy_path = os.path.join(base_model_dir, policy_file_name)
        policy_file_names.append(cfg.PPO.real_policy_path)
        cfg.PPO.wandb_run_name = f"env_{map_filename}_policy_{datetime.now().time()}"
        PPO_world_training.run_training_real_env(cfg)

        
    # load the map from MAP sample
    if cfg.training_generator.elites_path is not None:
        env_database = support.load_sample_MAP(cfg.training_generator.elites_path)
    else:
        env_text_file_name = ['env1_move.txt','env2_move.txt','env3_move.txt', 'env3_move.txt']
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'level'))
    if cfg.training_generator.elites_path is not None:
        step_len = len(env_database)
    else:
        step_len = len(env_text_file_name)

    for step in range(step_len):
        print(f"Step {step+1} of {step_len}...")

        # 1. load the generator  # just for the experiment, we load 2 env 11*11 with empty and wall and 3 env with key and door inside (anywhere)
        # model = support.load_gen_func()

        # 2. use the generator to generator env (comparision among the different env as loss1)
        # 1) by the generator
        # 2) by the env_database from MAP elites
        if cfg.training_generator.elites_path is not None:
            env = env_database[step]
            env = torch.tensor(env).unsqueeze(0)  # shape: (1, H, W)
            env = support.wrap_env(env)
            
        else:
            # env = support.wrap_env(support.generate_env(model))
            file_path = os.path.join(file_dir, env_text_file_name[step])
            env = support.wrap_env_from_text(file_path)

        # 3. collect data from the env and save it to npz
        support.del_env_data_file()  # clear the data_save_path
        support.collect_data_trainer(env)
        # support.visualize_dataset(cfg.attention_model.data_dir,'empty',200,True)

        # 4. train(finetuning) the attention & WM (according to the size of final task to padding the data)
        cur_old_params, cur_fisher = AttentionWM_training.train_api(cfg, old_params, fisher)
        old_params, fisher = cur_old_params, cur_fisher
        # for the second time, we do the finetuning based on the pre-trained model + the data collected from the env (EWC or LWF)

        # 5. using the trained attention & WM to play in the final task sets
        # load the final task env and train the PPO agent inside the learned world model
        # load all the finals in the task folder
        # the policy is training based on the env_path in PPO
        regret = []
        # policy_file_names = ['/home/siyao/project/rlPractice/MiniGrid/modelBased/models/PPO_gen_final_task_0.ckpt',
        #                      '/home/siyao/project/rlPractice/MiniGrid/modelBased/models/PPO_gen_final_task_1.ckpt',
        #                      '/home/siyao/project/rlPractice/MiniGrid/modelBased/models/PPO_gen_final_task_2.ckpt',
        #                      '/home/siyao/project/rlPractice/MiniGrid/modelBased/models/PPO_gen_final_task_3.ckpt',
        #                      '/home/siyao/project/rlPractice/MiniGrid/modelBased/models/PPO_gen_final_task_4.ckpt',]
        for i in range(len(final_task_set)):
            cfg.PPO.env_path = final_task_set[i]
            cfg.PPO.real_policy_path = policy_file_names[i]
            cfg.PPO.wandb_run_name = f"env_{step}_final_task_{i}_policy_{datetime.now().time()}"
            regret_cur = PPO_world_training.run_training_wm(cfg)
            regret.append(regret_cur)
        average_regret = sum(regret) / len(regret)
        
        # 6. return score in the final task 
        # evaluate the WM perfromance in final task (regret)
        if average_regret < cfg.PPO.regret_threshold:
            print('************************************************')
            print(f"Step {step}: Success! The average regret is {average_regret}")
            print('************************************************')
            break
        else:
            print('*************************************************')
            print(f'Step {step}: Fail! The average regret is {average_regret}')
            print('*************************************************')
        # clear the data_save_path
        support.del_env_data_file()

    
if __name__ == "__main__":
    run()