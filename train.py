#importing necessary libraries
import argparse
import json
from trainer_util import Trainer_Util

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Train script parser")

parser.add_argument ('data_dir', help = 'Data directory. Mandatory argument', type = str)
parser.add_argument ('--save_dir', help = 'Checkpoint saving directory', default='', type = str)
parser.add_argument ('--arch', help = 'Model for training', type = str, default = 'vgg16')
parser.add_argument ('--learning_rate', help = 'Learning rate', type = float, default='0.001')
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier', type = int, default='512')
parser.add_argument ('--epochs', help = 'Number of epochs', type = int, default = '10')
parser.add_argument ('--gpu', help = "Option to use GPU", action='store_true')

args = parser.parse_args()

trainer = Trainer_Util(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu, 'cat_to_name.json')
trainer.load_predict_save()