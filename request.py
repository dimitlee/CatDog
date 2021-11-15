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
cat_tp = 0
dog_tp = 0
for i in df.index:
    if df['ID'][i].split('.')[0] == 'cat' and df['cat_prob'][i] >= 0.5:
        cat_tp += 1
    if df['ID'][i].split('.')[0] == 'dog' and df['dog_prob'][i] > 0.5:
        dog_tp += 1
tp = pd.DataFrame({'cat_tp': [cat_tp], 'dog_tp': [dog_tp]})

writer = pd.ExcelWriter('./analysis.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Probabilities', index=False)
tp.to_excel(writer, sheet_name='Analysis', index=False)
writer.close()