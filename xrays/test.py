# %%
import glob, os
from torch.utils.data import Dataset, DataLoader
import torch, csv, pydicom
import torchvision.transforms as transforms
from PIL import Image
class RSNATestSet(Dataset):
    def __init__(self, files, root_dir):
        self.root_dir = root_dir
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.files[idx]
        img = pydicom.dcmread(file).pixel_array
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.ConvertImageDtype(torch.float),
        ])
        img = tf(Image.fromarray(img))
        return idx, img

# %%
import torch
from tqdm import tqdm
import torch.nn.functional as F
BATCH_SIZE = 128
base_path = '/home/datasets/rsna-intracranial-hemorrhage-detection/'
items = base_path + 'stage_2_test/'
dcms = glob.glob('{}*.dcm'.format(items))
testset = RSNATestSet(files=dcms, root_dir=items)
testloader = torch.utils.data.DataLoader(testset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
classes = ('epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'none')
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load('weights/resnet18/epoch4.pth'))
model.eval()
if torch.cuda.is_available():
    print("CUDA is available - using CUDA!")
    model = model.cuda()
predictions_output = {}
with tqdm(testloader, unit="batch") as tepoch:
    for name, inputs in tepoch:
        tepoch.set_description("Inference")
        # get the inputs; data is a list of [inputs, labels]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        # zero the parameter gradients

        # forward + backward + optimize
        outputs = F.softmax(model(inputs), dim=1)
        predictions = outputs.argmax(dim=1, keepdim=True).squeeze().tolist()
        name = name.tolist()
        for (idx, pred) in zip(name, predictions):
            predictions_output[dcms[idx]] = classes[pred]


