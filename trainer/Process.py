from omegaconf import DictConfig
import Support
from generator.common.utils import load_gen
from modelBased.common.utils import TRAINER_PATH
from modelBased import AttentionWM_training, PPO_world_training

import hydra
import os
import wandb

'''
Process
1. load the generator
2. use the generator to generator env 
(comparision among the different env as loss1)
3. collect data from the env
4. train(finetuning) the attention & WM
5. using the trained attention & WM to play in the final task sets
6. return score in the final task as the loss2
7. using loss1+loss2 update the generator

'''


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config")
def run(cfg: DictConfig):
    old_params, fisher = None, None

    support = Support.Support(cfg)
    # support.generate_final_task()

    env_text_file_name = ['env1_move.txt','env2_move.txt','env3_move.txt', 'env3_move.txt']
    file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'level'))
    for step in range(len(env_text_file_name)):
        print(f"Step {step+1} of 100")

        # add the global wandb
        # 1. load the generator  # just for the experiment, we load 2 env 11*11 with empty and wall and 3 env with key and door inside (anywhere)
        # model = support.load_gen_func()

        # 2. use the generator to generator env (comparision among the different env as loss1)
        # env = support.wrap_env(support.generate_env(model))
        file_path = os.path.join(file_dir, env_text_file_name[step])
        env = support.wrap_env_from_text(file_path)

        # 3. collect data from the env and save it to npz
        support.del_env_data_file()  # clear the data_save_path
        
        support.collect_data_trainer(env)
        support.visualize_dataset(cfg.attention_model.data_dir,'empty',200,True)
        # 4. train(finetuning) the attention & WM (according to the size of final task to padding the data)
        
        cur_old_params, cur_fisher = AttentionWM_training.train_api(cfg, old_params, fisher)
        old_params, fisher = cur_old_params, cur_fisher
        # for the second time, we do the finetuning based on the pre-trained model + the data collected from the env (EWC or LWF)

        # 5. using the trained attention & WM to play in the final task sets
        # load the final task env and train the PPO agent inside the learned world model
        
        PPO_world_training.run_training(cfg)
        # clear the data_save_path
        support.del_env_data_file()
        # 6. return score in the final task as the loss2


        # 7. using loss1+loss2 update the generator
    
    

    
    
    
    
    
if __name__ == "__main__":
    run()