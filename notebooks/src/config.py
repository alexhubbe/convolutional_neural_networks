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
OVERFIT_FIG = IMAGES_FOLDER/"overfit.jpg" # Figure depicting overfit
BETTERFIT_FIG = IMAGES_FOLDER/"betterfit.jpg" #Figure with a better fit
