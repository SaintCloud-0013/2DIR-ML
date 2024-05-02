from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from my_dataset import MyDataSet

def numpy_to_tensor(img):
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img

data_transform = {
    "train": transforms.Compose([numpy_to_tensor]),
    "val": transforms.Compose([numpy_to_tensor]),
}

def setup_dataloaders(args):
    dataset = MyDataSet(args.ir_file, args.ss_content_file)

    train_indices, val_indices = train_test_split(range(len(dataset)), train_size=args.train_size, random_state=42)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=lambda batch: tuple(zip(*((data_transform["train"](img), label, img_name) for img, label, img_name in batch))))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=lambda batch: tuple(zip(*((data_transform["val"](img), label, img_name) for img, label, img_name in batch))))

    return train_loader, val_loader