import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import vit_2dir as create_model

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
model = create_model(num_ss=3, has_logits=False).to(device)
model.load_state_dict(torch.load("./2dir_pre_trained.pth"))
model.eval()

npz_path = "./2dir.npz"

def load_npz_to_tensor(npz_path, device):
    data = np.load(npz_path)
    image_array = data['arr_0']
    if image_array.ndim == 2:
        image_array = image_array[np.newaxis, np.newaxis, :, :]
    image_tensor = torch.from_numpy(image_array).float().to(device)
    return image_tensor

image_tensor = load_npz_to_tensor(npz_path, device)

with torch.no_grad():
    _, attn_weights_list = model(image_tensor, return_attn_weights=True)

def attention_rollout(attn_weights_list, discard_ratio=0.9):
    rollout = torch.eye(attn_weights_list[0].size(-1) - 1).to(attn_weights_list[0].device)

    for attn in attn_weights_list:
        avg_attn = torch.mean(attn, dim=1)
        avg_attn = avg_attn[:, 1:, 1:]
        avg_attn = avg_attn.squeeze(0)
        
        rollout = torch.matmul(rollout, avg_attn)

    return rollout

rollout = attention_rollout(attn_weights_list)

def plot_attention_map(rollout, filename='attention_map.png'):
    rollout_np = rollout.cpu().detach().numpy()
    plt.imshow(rollout_np, cmap='jet')
    plt.colorbar()
    plt.title("Attention Map")

    plt.savefig(filename, dpi=300)
    plt.close()

plot_attention_map(rollout, 'attention_map.png')
