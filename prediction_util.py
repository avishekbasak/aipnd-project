import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms

#import helper
from collections import OrderedDict
from PIL import Image
import json
import warnings
import time
from predict_result import Predict_Result

class Prediction_Util:
    def __init__(self, image_path, checkpoint, top_k =1, category_names = None, gpu = True, num_classes =102):
        warnings.filterwarnings('ignore')
        
        self.image_path = image_path
        self.model = self.__load_checkpoint(checkpoint)
        self.top_k = top_k
        self.num_classes = num_classes
        self.category_names = self.__load_category_names(category_names)
        self.gpu = gpu and torch.cuda.is_available()
        
        
    def process_image(self, size_resize = 256, size_crop = 224, norm_mean = [0.485, 0.456, 0.406], norm_std = [0.229, 0.224, 0.225] ):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        image = Image.open(self.image_path)
        img_loader = transforms.Compose([transforms.Resize(size_resize),
                                         transforms.CenterCrop(size_crop), 
                                         transforms.ToTensor(),
                                         transforms.
                                         transforms.Normalize(norm_mean, norm_std)])

        if self.gpu:
            image = img_loader(image).float().cuda()
        else:
            image = img_loader(image).float()



        return image
    
    def __load_category_names(self, category_names):
        if category_names is None:
            return None
        with open(category_names, 'r') as f:
            return json.load(f)
        
    def __load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path)
        learning_rate = checkpoint['learning_rate']
        model = getattr(models, checkpoint['network'])(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.epochs = checkpoint['epochs']
        model.optimizer = checkpoint['optimizer']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    
        return model
    
    def predict(self):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        
        processed_image = self.process_image()
        processed_image.unsqueeze_(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        with torch.no_grad ():
            probabilities = torch.exp(self.model.forward(processed_image))


        top_probs, top_labs = probabilities.topk(self.top_k)

        mapping = {val: key for key, val in
                    self.model.class_to_idx.items()
                    }

        top_labels = []

        for label in top_labs[0].cpu().numpy():
            top_labels.append(mapping[label])

        top_flowers = [self.category_names[lab] for lab in top_labels]
    
        result= list()
        
        top_probs = top_probs[0].tolist()
        
        for idx in range(self.top_k):
            if self.category_names is None:
                result.append(Predict_Result(top_probs[idx], top_labels[idx], None))
            else:
                result.append(Predict_Result(top_probs[idx], top_labels[idx], top_flowers[idx]))
                
        return result