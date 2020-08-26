import math
import torch
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms

#import helper
from collections import OrderedDict
from PIL import Image
import json
import warnings
import time


class Trainer_Util:
    def __init__(self, data_dir, save_dir, arch ='vgg16', learning_rate = 0.001, hidden_units='512', epochs=10, gpu = True, cat_to_name_path = 'cat_to_name.json'):
        warnings.filterwarnings('ignore')
        
        self.arch = arch
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.gpu = gpu and torch.cuda.is_available()
        self.category_names = self.__load_category_names(cat_to_name_path)
        self.output_size = len(self.category_names)
        self.learning_rate = learning_rate
        self.model = self.__load_model(arch, hidden_units)
        self.epochs = epochs
        self.batch_size = 64
    
    def __load_category_names(self, category_names):
        if category_names is None:
            return None
        with open(category_names, 'r') as f:
            return json.load(f)
    
    def __load_model(self, arch, hidden_units, drop_rate = 0.3):
        model = getattr(models, arch)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        feature_num = model.classifier[0].in_features
        in_netween_num = math.floor((feature_num + hidden_units) / 4)
        
        model.classifier = nn.Sequential(OrderedDict([
                     ('fc1', nn.Linear(feature_num, in_netween_num)),
                     ('relu1', nn.ReLU()),
                     ('dropout1', nn.Dropout(p = drop_rate)),
                     ('fc2',nn.Linear(in_netween_num, hidden_units)),
                     ('relu2', nn.ReLU()),
                     ('dropout2', nn.Dropout(p = drop_rate)),
                     ('fc3', nn.Linear(hidden_units, self.output_size)),
                     ('output', nn.LogSoftmax(dim=1))
                     ]))
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
        if self.gpu :
            model.to('cuda')
        
        return model
    
    def load_predict_save(self):
        self.__load_data()
        self.__train()
        self.__save_checkpoint()
    
    def __load_data(self):
        data_dir = self.data_dir
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        
        degrees_rotation = 30
        size_crop = 224
        size_resize = 255
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        batch_size = 64

        data_transforms = {
            'train' : transforms.Compose([transforms.RandomRotation(degrees_rotation),
                                               transforms.RandomResizedCrop(size_crop),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(norm_mean, norm_std)]) ,

            'valid' : transforms.Compose([transforms.Resize(size_resize), 
                                               transforms.CenterCrop(size_crop),
                                               transforms.ToTensor(),
                                               transforms.Normalize(norm_mean, norm_std)
                                              ])
        }

        self.image_datasets = {
            'train' : datasets.ImageFolder(train_dir, transform = data_transforms['train']),
            'valid' : datasets.ImageFolder(valid_dir, transform = data_transforms['valid'])
        }
        
        self.dataloaders = {
            'train' : torch.utils.data.DataLoader(self.image_datasets['train'], batch_size = batch_size, shuffle = True),
            'valid' : torch.utils.data.DataLoader(self.image_datasets['valid'], batch_size = batch_size)
        }
        
    def __save_checkpoint(self):
        print('Save Checkpoint')
        checkpoint = {'network': self.arch,
              'output_size': self.output_size,
              'learning_rate': self.learning_rate,       
              'batch_size': self.batch_size,
              'classifier' : self.model.classifier,
              'epochs': self.epochs,
              'optimizer': self.optimizer.state_dict(),
              'state_dict': self.model.state_dict(),
              'class_to_idx': self.image_datasets['train'].class_to_idx}
        file = self.save_dir+'checkpoint.pth'
        torch.save(checkpoint, self.save_dir+'checkpoint.pth')
        print('Save checkpoint to: '+ file)
    
    def __train(self):
        steps = 0
        running_loss = 0
        print_every = 40
        start_time = time.time()
        val_loss = []
        train_loss = []
        print('Training Started')
        for epoch in range(self.epochs):
            for inputs, labels in self.dataloaders['train']:
                steps += 1
                # Move input and label tensors to the default device
                if self.gpu:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                #reset optimizer
                self.optimizer.zero_grad()
                #forward pass
                log_ps = self.model(inputs)
                #loss calculation
                loss = self.criterion(log_ps, labels)
                #back propoagation
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.dataloaders['valid']:
                            if self.gpu:
                                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                            logps = self.model.forward(inputs)
                            batch_loss = self.criterion(logps, labels)

                            validation_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    valid_len = len(self.dataloaders['valid'])      
                    train_loss.append(running_loss/print_every)
                    val_loss.append(validation_loss/valid_len)
                    print(f"Epoch {epoch+1}/{self.epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {validation_loss/valid_len:.3f}.. "
                          f"Validation accuracy: {accuracy/valid_len*100:.3f}%")
                    running_loss = 0
                    self.model.train()

        end_time = time.time()

        print('\nTraining time: ' + str(end_time - start_time) +'s')