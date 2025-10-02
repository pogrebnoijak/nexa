import os

MINUTE = 60
FISHER_MINIMUM = 10 * 60  # 10 min
FISHER_MAX_WINDOW = 30 * 60  # 30 min
INF = 1e6
ZERO_PLUS = 1e-6
RELIABLE_K = 0.5
SLEEP_WS = 10

APP_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
SHORT_ASSET_PATH_F = os.path.join(APP_DIR_PATH, "assets", "short_%d.pkl")
SHORT_CNN_ASSET_PATH_F = os.path.join(APP_DIR_PATH, "assets", "cnn_gru_%d.pth")
LONG_ASSET_PATH = os.path.join(APP_DIR_PATH, "assets", "long.pkl")
DATA_PATH_PREFIX = os.path.join(APP_DIR_PATH, "data")
DATA_HYPOXIA_PATH = os.path.join(DATA_PATH_PREFIX, "hypoxia")
DATA_REGULAR_PATH = os.path.join(DATA_PATH_PREFIX, "regular")

IS_DEMO = os.environ.get("IS_DEMO", "True") == "True"
SCORE_BOUNDARY = float(os.environ.get("SCORE_BOUNDARY", "0.5"))
DEVICE = os.environ.get("DEVICE", "cpu")  # cpu or gpu

__PREDICTION_SHORT_WINDOW_SIZES = [
    int(x)
    for x in os.environ.get("PREDICTION_SHORT_WINDOW_SIZES", "10:20:40").split(":")
]  # 10, 20, 40 min
__PREDICTION_SHORT_WINDOW_SIZES_CNN = [
    int(x)
    for x in os.environ.get("PREDICTION_SHORT_WINDOW_SIZES_CNN", "10:20:40").split(":")
]  # 10, 20, 40 min
__PREDICTION_SHORT_STEPS = [
    int(x) for x in os.environ.get("PREDICTION_SHORT_STEPS", "1:1:1").split(":")
]  # 1 min
__PREDICTION_SHORT_HORIZONS = [
    int(x) for x in os.environ.get("PREDICTION_SHORT_HORIZONS", "10:20:40").split(":")
]  # 10, 20, 40 min
# horizon -> (window, step)
PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP = {
    horizon: (__PREDICTION_SHORT_WINDOW_SIZES[i], __PREDICTION_SHORT_STEPS[i])
    for i, horizon in enumerate(__PREDICTION_SHORT_HORIZONS)
}
PREDICTION_SHORT_HORIZON_TO_WINDOW_SIZE_AND_STEP_CNN = {
    horizon: (__PREDICTION_SHORT_WINDOW_SIZES_CNN[i], __PREDICTION_SHORT_STEPS[i])
    for i, horizon in enumerate(__PREDICTION_SHORT_HORIZONS)
}
PREDICTION_SHORT_CNN_EPOCHS = int(os.environ.get("PREDICTION_SHORT_CNN_EPOCHS", "10"))

PREDICTION_LONG_WINDOW_SIZE = int(
    os.environ.get("PREDICTION_LONG_WINDOW_SIZE", "20")
)  # 20 min
PREDICTION_LONG_STEP = int(os.environ.get("PREDICTION_LONG_STEP", "5"))  # 5 min
