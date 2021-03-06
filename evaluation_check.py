import requests
import os
import csv
from tqdm import tqdm
path = '/data1/CIM/Test_API/check/'

with open('check_signature.csv', mode='w') as compare_result_file:
    csv_writer = csv.writer(
        compare_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(
        ['file name', 'class'])
    for file in tqdm(os.listdir(path)):
        files = []
        files.append(('file', open(path+file, 'rb')))
        r = requests.post(
            'http://localhost:6666/check_signature/', files=files)
        data = r.json()
        csv_writer.writerow([file, data['status']])
