import json
import base64
import requests
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--port',
    type=int,
    default=8000,
    help='The port that the server is running on'
)
parser.add_argument(
    '--directory',
    type=str,
    default='data',
    help='The directory that stores cat and dog images'
)
args = parser.parse_args()

url = f'http://127.0.0.1:{args.port}/predict'
directory = f'./{args.directory}'
input = {'photos': []}

for image in os.listdir(directory):
    with open(f'{directory}/{image}', 'rb') as image_file:
        encoding = base64.b64encode(image_file.read()).decode('utf-8')
        image_dict = {'ID': image, 'img_code': encoding}
        input['photos'].append(image_dict)

response = requests.post(url, data=json.dumps(input))

df = pd.DataFrame(response.json()['results'])
df.to_excel('./analysis.xlsx')