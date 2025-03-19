import os


class Paths:
    ROOT = os.path.dirname(__file__)
    LEVEL_FILE = os.path.join(ROOT, "level//Grid_8_8_empty.txt")
    LEVEL_FILE_Rmax = os.path.join(ROOT, "level/empty.txt")
    LEVEL_FILE_Rmax2 = os.path.join(ROOT, "level/Grid_21_21_empty.txt")
    EXPERIMENT_RESULT = os.path.join(ROOT, 'experiment_MF/result')
    EXPERIMENT_VISUALIZED = os.path.join(ROOT, 'experiment_MF/visualize')

    # model_based learning path
    EXPERIMENT_RESULT_MB = os.path.join(ROOT, 'experiment_MF/result')
    EXPERIMENT_VISUALIZED_MB = os.path.join(ROOT, 'experiment_MF/visualize')
    MODEL_BASED_DATA = os.path.join(ROOT, 'modelBased/data')
    CHARACTOR_DATA = os.path.join(ROOT, 'modelBased/data/env_charactor')
    TRAINED_MODEL = os.path.join(ROOT, 'modelBased/model')