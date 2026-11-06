from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
# input files
HUMAN_CSV = DATA_DIR / "human_text.csv"
AI_CSV = DATA_DIR / "ai_text.csv"
MODIFIED_CSV = DATA_DIR / "modified_ai_text.csv"   # optional
# files we create
PROCESSED_CSV = DATA_DIR / "processed_dataset.csv"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
PROPOSED_MODEL_PATH = MODELS_DIR / "proposed_model.pkl"
FEATURE_REFERENCE_PATH = MODELS_DIR / "feature_reference.pkl"
METRICS_TXT = RESULTS_DIR / "metrics.txt"
ALL_RESULTS_CSV = RESULTS_DIR / "all_results.csv"
CONFUSION_PNG = RESULTS_DIR / "confusion_matrix.png"
COMPARISON_PNG = RESULTS_DIR / "model_comparison.png"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MIN_WORDS = 5            #check if the number of words is greter than the given value.
MIN_WORDS_PREDICT = 5
WARN_WORDS_PREDICT = 30
C_GRID = [0.1, 1.0, 10.0]

class ProjectError(Exception):
    pass
def ensure_dirs():
    for folder in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
