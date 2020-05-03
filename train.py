import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import NotaDataset, collate_fn
import model
from tqdm import tqdm

#Hyper-parameters
num_classes = 6
learning_rate = 0.0005
weight_decay = 0.01
num_epochs = 30

#Loading dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = NotaDataset(root="data", train=True, transform=transform)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

#Setting a device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Loading our model
model = model.get_model(num_classes)

#Moving our model into the device
model.to(device)

# Parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(
    params, lr=learning_rate, weight_decay=weight_decay
)

# Training
for epoch in range(num_epochs):
    print("# Training the facial emotions recognition model")
    print(f"Epoch: {epoch}/{num_epochs}")
    model.train()
    i = 0
    for imgs, annotations in tqdm(train_dataloader):

        i += 1

        imgs = list(img.to(device) for img in imgs)
        #for a half precision model
        #imgs = list(img.to(device, torch.float16) for img in imgs)

        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Train Loss: {losses:.4f}")

    torch.save(model.state_dict(), f'checkpoints/epoch-{epoch}.pth')

