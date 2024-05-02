import torch
from model import vit_2dir as create_model

def initialize_model(args, device):
    model = create_model(num_ss=3, has_logits=False).to(device)

    if args.weights:
        model_weights = torch.load(args.weights)
        model.load_state_dict(model_weights)

    if args.freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

    return model, optimizer, scheduler
