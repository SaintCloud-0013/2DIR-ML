import torch

def train_one_epoch(model, train_loader, optimizer, scheduler, device, tb_writer, epoch):
    model.train()
    train_loss = 0.0

    for batch_idx, (inputs, labels, img_names) in enumerate(train_loader):
        inputs = torch.stack(inputs).float().to(device)
        labels = torch.stack(labels).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.MSELoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_data["images"].extend(img_names)
        train_data["labels"].extend(labels.cpu().numpy())
        train_data["outputs"].extend(outputs.detach().cpu().numpy())

        scheduler.step()

    train_loss /= len(train_loader)
    tb_writer.add_scalar('Train/Loss', train_loss, epoch)

    return train_loss