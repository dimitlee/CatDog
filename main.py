import base64
import sys
import warnings
import io
import torch
from torchvision import transforms
from flask import Flask, request, jsonify
from PIL import Image
from utils.helpers import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--port',
    type=int,
    default=8000,
    help='Set the port for the server to run on'
)
args = parser.parse_args()

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings('ignore')

model = load_model('./models/catvdog.pth')
model.eval()
model.to(device)


'''
    decode image from base64 format and apply image preprocessing
'''
def base64_image_transform(image):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image_bytes = base64.b64decode(image)
    image = Image.open(io.BytesIO(image_bytes))
    imagetensor = test_transforms(image)
    return imagetensor

'''
    make predictions
'''
@app.route('/predict', methods=['POST'])
def predict():
    output = {'results': []}
    data = request.get_json(force=True)
    for i in request.json['photos']:
        id = i['ID']
        image = base64_image_transform(i['img_code'])
        image1 = image[None,:,:,:].to(device)
        ps=torch.exp(model(image1))
        img_dict = {'ID': id,
                    'cat_prob': ps[0][0].item(),
                    'dog_prob': ps[0][1].item()}
        output['results'].append(img_dict)
    
    return jsonify(output), 200


if __name__ == '__main__':
    app.run(port=args.port, debug=False)