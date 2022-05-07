# %%
base_path = '/home/datasets/rsna-intracranial-hemorrhage-detection/'
csv_path = base_path + 'stage_2_train.csv'
items = base_path + 'stage_2_train/'

# %%
from torch.utils.data import Dataset, DataLoader
import torch, csv, pydicom
import torchvision.transforms as transforms
from PIL import Image

class RSNASet(Dataset):
    def __init__(self, csv_path, root_dir):
        self.root_dir = root_dir
        labels = {}
        with open(csv_path) as f:
            rdr = csv.reader(f)
            next(rdr)
            for rows in rdr:
                fn = '_'.join(rows[0].split('_')[0:2])
                try:
                    labels[fn].append(int(rows[1]))
                except:
                    labels[fn] = [int(rows[1])]
        for fn in labels.keys():
            # "any" should really be the classification "none"
            # so, it should only be 1 if nothing else is true - swap the values
            labels[fn][5] = abs(labels[fn][5]-1)
        self.labels = labels
        self.files = list(labels.keys())
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.files[idx]
        label = torch.argmax(torch.FloatTensor(self.labels[file]))
        img = pydicom.dcmread(self.root_dir+file+'.dcm').pixel_array
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.ConvertImageDtype(torch.float),
        ])
        img = tf(Image.fromarray(img))
        return img, label

# %%
tempset = RSNASet(csv_path=csv_path, root_dir=items)

# %%
BATCH_SIZE = 128

import math
trainlen = math.floor(0.80 * len(tempset)) # 80/20 train/val split
vallen = len(tempset) - trainlen
# trainlen = 512
# vallen = 256

trainset, valset = torch.utils.data.random_split(tempset, [trainlen, vallen], generator=torch.Generator().manual_seed(42))
model = 'resnet18'
trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
valloader = torch.utils.data.DataLoader(valset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
classes = ('epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural','none')
model = torch.hub.load('pytorch/vision:v0.10.0', model, pretrained=False, num_classes=len(classes))
if torch.cuda.is_available():
    print("CUDA is available - using CUDA!")
    model = model.cuda()
import torch.optim as optim
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# %%

from tqdm import tqdm
for epoch in range(5):  # loop over the dataset multiple times
    with tqdm(trainloader, unit="batch") as tepoch:
        model.train()
        running_loss = 0.0
        batch = 0
        total_correct = 0
        total = 0
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total += BATCH_SIZE
            accuracy = total_correct / total # batch_size
            running_loss += loss.item()
            if batch > 0:
                tepoch.set_postfix(loss=running_loss/batch, accuracy=100. * accuracy)
            batch += 1
    with tqdm(valloader, unit="batch") as tepoch:
        model.eval()
        running_loss = 0.0
        batch = 0
        total_correct = 0
        total = 0
        for inputs, labels in tepoch:
            tepoch.set_description(f"Validating: Epoch {epoch + 1}")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total += BATCH_SIZE
            accuracy = total_correct / total # batch_size
            running_loss += loss.item()
            if batch > 0:
                tepoch.set_postfix(loss=running_loss/batch, accuracy=100. * accuracy)
            batch += 1

    print('Saving model at epoch {}'.format(epoch))
    torch.save(model.state_dict(), "weights/split_resnet18/epoch{}.pth".format(epoch))
    print('Saved model at epoch {}!'.format(epoch))
print('Finished training!')


