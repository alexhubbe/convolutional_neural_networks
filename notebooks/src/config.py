from pathlib import Path

PALETTE = 'cividis'
RANDOM_STATE = 42
SCATTER_ALPHA = 0.2


PROJECT_FOLDER = Path(__file__).resolve().parents[2]
# put the path for the project model files below
MODELS_FOLDER = PROJECT_FOLDER / "models"

AUTOGLUON_MODEL = MODELS_FOLDER/"autogluonmodel" #path to save the results from AutoMM (see 02_ah_autogluon.ipynb)
MODELCHECKPOINT = MODELS_FOLDER/"best_model.keras" #stores the best-performing model during training based on validation sparse categorical accuracy in MobileNet Model II.5.
KT_TUNER_DIRECTORY = MODELS_FOLDER/"kt_mobilenet_tuning" #folder to store Kera's hyperparameters optimization
KT_TUNER_PROJECT_NAME = MODELS_FOLDER/KT_TUNER_DIRECTORY/"mobilenet_bayesian" #folder with the trials from the optimization procedure.

# put any other necessary paths below
REPORT_FOLDER = PROJECT_FOLDER / "reports"
IMAGES_FOLDER = REPORT_FOLDER / "images"

README_INTRO_FIG = IMAGES_FOLDER/"example_images.jpg" # Figure to Illustrate the Pictures within Each Class
OVERFIT_FIG = IMAGES_FOLDER/"overfit.jpg" # Figure depicting overfit in the train and validation loss curve
BETTERFIT_FIG = IMAGES_FOLDER/"betterfit.jpg" #Figure with a better fit in the train and validation loss curve


from pathlib import Path

# Constants for visualization and reproducibility
PALETTE = 'cividis'
RANDOM_STATE = 42
SCATTER_ALPHA = 0.2

# Define project folder paths
PROJECT_FOLDER = Path(__file__).resolve().parents[2]

# Path for model files
MODELS_FOLDER = PROJECT_FOLDER / "models"

# Paths for specific model files and directories
AUTOGLUON_MODEL = MODELS_FOLDER / "autogluonmodel"  # Path to save results from AutoMM (see 02_ah_autogluon.ipynb)
MODELCHECKPOINT = MODELS_FOLDER / "best_model.keras"  # Stores the best-performing model during training based on validation sparse categorical accuracy (used in MobileNet Model II.5)
KT_TUNER_DIRECTORY = MODELS_FOLDER / "kt_mobilenet_tuning"  # Directory to store Keras hyperparameter optimization results
KT_TUNER_PROJECT_NAME = KT_TUNER_DIRECTORY / "mobilenet_bayesian"  # Directory for trials from the optimization procedure

# Additional necessary paths
REPORT_FOLDER = PROJECT_FOLDER / "reports"
IMAGES_FOLDER = REPORT_FOLDER / "images"

# Paths for report images
README_INTRO_FIG = IMAGES_FOLDER / "example_images.jpg"  # Figure illustrating example images within each class
OVERFIT_FIG = IMAGES_FOLDER / "overfit.jpg"  # Figure depicting overfitting
BETTERFIT_FIG = IMAGES_FOLDER / "betterfit.jpg"  # Figure showing a better fit
