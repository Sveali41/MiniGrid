from omegaconf import DictConfig
import Support
from generator.common.utils import load_gen
from modelBased.common.utils import TRAINER_PATH
import hydra
from learning_buffer import EnvLearningBuffer
import numpy as np
import wandb
from fisher_buffer import FisherReplayBuffer



'''
Process
1. load the generator (MAP selected env)
2. use the generator to generator env 
3. main function 
    if first iteration:
        train WM on the first env
        edit this env
        evaluation the WM on the edited env
        whether or not put it in the learning buffer
    else:
        decition model D(d): 1/0
        if d == 0:
            generator new env & collect data
            validation the WM performance on the new env
            whether or not put it in the learning buffer
        if d == 1:
            load the env from the learning buffer 
            train WM on the env
            edit this env
            evaluate the WM on the edited env
            whether or not put it in the learning buffer

'''


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def run(cfg: DictConfig):
    use_wandb = cfg.training_generator.use_wandb
    if use_wandb:
        wandb.login(key="eeecc8f761c161927a5713203b0362dfcb3181c4")
        wandb.init(project='World_Model_Curriculum_Learning', entity='18920011663-king-s-college-london', reinit=True)
    old_params, fisher = None, None
    fisher_buffer = FisherReplayBuffer(max_size=150000)
    learning_steps = cfg.training_generator.learning_steps
    support = Support.Support(cfg)
    # load the map from MAP sample
    print("++++++++++++++++++++++++++++++++++++  loading tasks... ++++++++++++++++++++++++++++++++++++++++++++++")
    env_database, file_dir = support.loading_tasks(cfg)
    learning_buffer = EnvLearningBuffer(max_size=cfg.training_generator.learning_buffer_size)

    for step in range(learning_steps):
        # within certain training steps
        print(f"Step {step+1} of {learning_steps}...")
        decision =  support.decision_model()
        if len(fisher_buffer) > 0:
            replay_data = fisher_buffer.export_dict()
        else:
            replay_data = None
            
        # === Step 1: 决定 env 来源 ===
        if step == 0 or len(learning_buffer) == 0:
            env, env_layout = support.generate_env_from_generator(
                cfg, env_database[step], file_dir
            )
            cfg.attention_model.freeze_weight = False
            support.collect_data_from_env(env, validate=cfg.attention_model.freeze_weight)
            cur_old_params, cur_fisher = support.train_world_model(cfg, old_params, fisher, env_layout=None, replay_data=replay_data)
            old_params, fisher = cur_old_params, cur_fisher
            task_npz_train = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples_train = {
                'obs': task_npz_train['a'],
                'obs_next': task_npz_train['b'],
                'act': task_npz_train['c'],
            }
            fisher_buffer.update_with_random_by_ratio(samples_train, 0.3)
            print("+++++++++++ Editing env +++++++++++")
            env_edited, env_layout = support.env_editor(env_layout, cfg.training_generator.dynamic_objects)
            print("+++++++++++ Checking if add to buffer +++++++++++")
            cfg.attention_model.freeze_weight = True
            support.collect_data_from_env(env_edited, validate=cfg.attention_model.freeze_weight)
            task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples = {
                'obs': task_npz['a'],
                'obs_next': task_npz['b'],
                'act': task_npz['c'],
            }
            wm_loss = support.validate_world_model(cfg, old_params, fisher, env_edited)
            support.add_into_learning_buffer(env_layout, wm_loss, samples, learning_buffer)

        elif decision == 0:
            cfg.attention_model.freeze_weight = False
            env, env_layout = support.generate_env_from_generator(
                            cfg, env_database[step], file_dir
                        )
            cfg.attention_model.freeze_weight = True
            support.collect_data_from_env(env, validate=cfg.attention_model.freeze_weight)
            task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples = {
                'obs': task_npz['a'],
                'obs_next': task_npz['b'],
                'act': task_npz['c'],
            }
            wm_loss = support.validate_world_model(cfg, old_params, fisher, env_layout)
            support.add_into_learning_buffer(env_layout, wm_loss, samples, learning_buffer)

        elif decision == 1:
            cfg.attention_model.freeze_weight = False
            # load the env from the learning buffer
            env, env_string = support.load_env_from_buffer(learning_buffer)
            support.collect_data_from_env(env, validate=cfg.attention_model.freeze_weight)
            cur_old_params, cur_fisher = support.train_world_model(cfg, old_params, fisher, env_layout=None, replay_data=replay_data)
            task_npz_train = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples_train = {
                'obs': task_npz_train['a'],
                'obs_next': task_npz_train['b'],
                'act': task_npz_train['c'],
            }
            fisher_buffer.update_with_random_by_ratio(samples_train, 0.4)
            old_params, fisher = cur_old_params, cur_fisher
            print("+++++++++++ Editing env +++++++++++")
            env_edited, env_layout = support.env_editor(env_string, cfg.training_generator.dynamic_objects)
            print("+++++++++++ Checking if add to buffer +++++++++++")
            cfg.attention_model.freeze_weight = True
            support.collect_data_from_env(env_edited, validate=cfg.attention_model.freeze_weight)
            task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
            samples = {
                'obs': task_npz['a'],
                'obs_next': task_npz['b'],
                'act': task_npz['c'],
            }
            wm_loss = support.validate_world_model(cfg, old_params, fisher, env_edited)
            support.add_into_learning_buffer(env_layout, wm_loss, samples, learning_buffer)
            

        if step % 10 == 0 and step != 0:
            rows = 30
            cols = 30
            num_maps = 5
            final_task_set = support.generate_final_task_set(rows, cols, num_maps)
            # === Step 2: Assessing performance on final task set ===
            avg_loss = support.assessing_performance_on_final_task(cfg, final_task_set)
            support.train_policy_on_final_task(cfg, final_task_set)
            # train the policy on the final task set
            if use_wandb:
                wandb.log({"final_task_performance": float(avg_loss)}, step=step)
    if use_wandb:
        wandb.finish()
    
if __name__ == "__main__":
    run()