import os
import pickle
from tensorboardX import SummaryWriter

def setup_logging():
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_writer = SummaryWriter(log_dir)

    model_save_path = "./weights"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    epoch_data_path = "./epoch_raw_data"
    if not os.path.exists(epoch_data_path):
        os.makedirs(epoch_data_path)

    return tb_writer, model_save_path, epoch_data_path

def save_epoch_data(epoch, train_data, val_data, epoch_data_path, min_val_loss, current_val_loss):
    if current_val_loss < min_val_loss:
        train_data_path = os.path.join(epoch_data_path, f"train_data_epoch_{epoch + 1}.pickle")
        val_data_path = os.path.join(epoch_data_path, f"val_data_epoch_{epoch + 1}.pickle")

        with open(train_data_path, "wb") as f:
            pickle.dump(train_data, f)
        
        with open(val_data_path, "wb") as f:
            pickle.dump(val_data, f)

        print(f"Data from epoch {epoch + 1} saved: validation loss improved.")