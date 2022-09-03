import torch
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# user created modules
from utils import save_checkpoint,load_checkpoint,print_examples
from get_loader import get_loader
from model import CNNtoRNN

def save_logs(epoch,loss_val):
    log_path = 'logs/logs.csv'
    dicti = {"epoch": epoch, "loss": loss_val}
    log = pd.read_csv(log_path)
    log = log.append(dicti,ignore_index=True)
    log.to_csv(log_path,index=False)

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = True
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter('runs/flicker')
    step = 0  # tensorboard visualisation step

    # intialise model,loss etc
    model = CNNtoRNN(embed_size,hidden_size,vocab_size,num_layers).to(device)
    criterian = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    # only fintune the last layer of CNN
    for name,param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load('my_checkpoint.pth.tar'),model,optimizer)
    
    model.train()

    # 52 done
    for epoch in range(44,52):

        print(epoch)
        if save_model:
            check_point = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(check_point)

        for idx,(imgs,captions) in \
                tqdm(enumerate(train_loader),total = len(train_loader),leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # leaving end token out of the captions
            outputs = model(imgs,captions[:-1])
            outputs = outputs.reshape(-1,outputs.shape[2])
            captions = captions.reshape(-1)
            loss = criterian(
                outputs,captions
            )
           
            # writing in log files/tensorboard
            loss_val = loss.cpu().detach().numpy()
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step+=1    


            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        # saving loss and epoch in logs
        save_logs(epoch,loss_val)
        
    
if __name__ == "__main__":
    train()

