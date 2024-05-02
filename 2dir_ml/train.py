import argparse
import torch
from datamodules import setup_dataloaders
from modules import setup_logging, initialize_model, train_one_epoch, validate_one_epoch

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else torch.device('cpu'))

    tb_writer, model_save_path = setup_logging()

    train_loader, val_loader = setup_dataloaders(args)

    model, optimizer, scheduler = initialize_model(args, device)

    min_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, tb_writer, epoch)

        val_loss = validate_one_epoch(model, val_loader, device, tb_writer, epoch)

        if val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_save_path}/model_best_val_loss.pth')

    tb_writer.close()
    print("Training completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--ir_file', type=str, default='./2dir.npz')
    parser.add_argument('--ss_content_file', type=str, default='./ss_content.csv')

    args = parser.parse_args()

    main(args)
