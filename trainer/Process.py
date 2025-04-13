from omegaconf import DictConfig
import Support
from generator.common.utils import load_gen
from modelBased.common.utils import TRAINER_PATH
import hydra
import os

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
    support = Support.Support(cfg)
    # 1. load the generator
    model = support.load_gen_func()

    # 2. use the generator to generator env (comparision among the different env as loss1)
    env = support.wrap_env(support.generate_env(model))

    # 3. collect data from the env
    support.collect_data(env)
    
    # 4. train(finetuning) the attention & WM (according to the size of final task to padding the data)
    # for the first time, we do the finetuning based on the pre-trained model
    
    # for the second time, we do the finetuning based on the pre-trained model + the data collected from the env (EWC or LWF)


    # 5. using the trained attention & WM to play in the final task sets


    # 6. return score in the final task as the loss2


    # 7. using loss1+loss2 update the generator
    
    

    
    
    
    
    
if __name__ == "__main__":
    run()