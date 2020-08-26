import argparse
from  prediction_util import Prediction_Util
from predict_result import Predict_Result


parser = argparse.ArgumentParser(description='Flower Classifier')
parser.add_argument("image_path", help='Path of the image that needs to be classified')
parser.add_argument("checkpoint", help='Model that will be used for classification')
parser.add_argument('--top_k', type=int, default='3',
    help='Return the top KK most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json',
    help='Path to a JSON file mapping labels to flower names')
parser.add_argument ('--gpu', help = "Option to use GPU", action='store_true')

args = parser.parse_args()

prediction_util = Prediction_Util(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)

result = prediction_util.predict()

for r in result:
    print(r)    