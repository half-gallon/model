import os
from load_training_data import load_training_data
from read_dir import read_dir
from model import train_model

DATA_DIR = "data"
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")


os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)

########################################################################
# Load files
########################################################################


TRAIN_DATA_FILES = read_dir(TRAIN_DATA_DIR)
TEST_DATA_FILES = read_dir(TEST_DATA_DIR)

print(
    f"Training data - {len(TRAIN_DATA_FILES)}",
)
print(
    f"Test data     -  {len(TEST_DATA_FILES)}",
)


########################################################################
# Load data
########################################################################


dataloader, max_time_dim = load_training_data(
    TRAIN_DATA_DIR, batch_size=len(TEST_DATA_FILES)
)


model = train_model(dataloader, num_epochs=2)
