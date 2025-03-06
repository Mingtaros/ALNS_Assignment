import os

# Directories
MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(MAIN_DIR, os.pardir))
SCRIPT_DIR = os.path.dirname(PARENT_DIR)

INPUT = os.path.join(MAIN_DIR, "input")
OUTPUT = os.path.join(MAIN_DIR, "output")
RESULT = os.path.join(MAIN_DIR, "result")
DATA_PATH = os.path.join(PARENT_DIR, "psp_instances")
TRAINED_MODELS = os.path.join(MAIN_DIR, "trained_models")
CONFIG = os.path.join(MAIN_DIR, "dr_configs")
