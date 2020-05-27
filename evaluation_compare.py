import requests
import os
import csv
from tqdm import tqdm
path = '/data1/CIM/Test_API/compare/'

with open('compare_result_signature.csv', mode='w') as compare_result_file:
    csv_writer = csv.writer(
        compare_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(
        ['file name', 'status', 'similarity', 'app_class', 'device_class'])
    for file in tqdm(set(os.listdir(path + 'app/') + os.listdir(path + 'device/'))):
        files = []
        try:
            files.append(('app', open(path + 'app/' + file, 'rb')))
        except:
            files.append(('app', b''))
        try:
            files.append(('device', open(path + 'device/' + file, 'rb')))
        except:
            files.append(('device', b''))
        r = requests.post(
            'http://localhost:6666/compare_signature/', files=files)
        data = r.json()
        if(data['status'] != "error"):
            csv_writer.writerow(
                [file, data['status'], data['similarity'], data['app_class'], data['device_class']])
        else:
            csv_writer.writerow(
                [file, data['status'], '', '', ''])
