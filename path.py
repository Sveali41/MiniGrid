import os


class Paths:
    ROOT = os.path.dirname(__file__)
    LEVEL_FILE = os.path.join(ROOT, "level/final_env.txt")
    EXPERIMENT_RESULT = os.path.join(ROOT, 'experiment_MF/result')
    EXPERIMENT_VISUALIZED = os.path.join(ROOT, 'experiment_MF/visualize')

    # model_based learning path
    EXPERIMENT_RESULT_MB = os.path.join(ROOT, 'experiment_MF/result')
    EXPERIMENT_VISUALIZED_MB = os.path.join(ROOT, 'experiment_MF/visualize')
    MODEL_BASED_DATA = os.path.join(ROOT, 'modelBased/data')
    CHARACTOR_DATA = os.path.join(ROOT, 'modelBased/data/env_charactor')
    TRAINED_MODEL = os.path.join(ROOT, 'modelBased/model')