import os


class Paths:
    ROOT = os.path.dirname(__file__)
    LEVEL_FILE = os.path.join(ROOT, "level/simple_test_corridor_long.txt")
    EXPERIMENT_RESULT = os.path.join(ROOT, 'experiment/result')
    EXPERIMENT_VISUALIZED = os.path.join(ROOT, 'experiment/visualize')
