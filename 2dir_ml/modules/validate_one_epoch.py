import torch

def validate_one_epoch(model, val_loader, device, tb_writer, epoch):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels, img_names) in enumerate(val_loader):
            inputs = torch.stack(inputs).float().to(device)
            labels = torch.stack(labels).to(device)

            outputs = model(inputs)
            loss = torch.nn.MSELoss()(outputs, labels)
            val_loss += loss.item()

            val_data["images"].extend(img_names)
            val_data["labels"].extend(labels.cpu().numpy())
            val_data["outputs"].extend(outputs.detach().cpu().numpy())    val_loss /= len(val_loader)

    tb_writer.add_scalar('Val/Loss', val_loss, epoch)

    return val_loss