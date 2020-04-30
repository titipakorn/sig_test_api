import requests
import os
import csv
path = '/data1/CIM/Test_API/compare/'

with open('compare_result_signature.csv', mode='w') as compare_result_file:
    csv_writer = csv.writer(
        compare_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for file in os.listdir(path+'app/'):
        files = []
        files.append(('app', open(path+'app/'+file, 'rb')))
        files.append(('device', open(path+'device/'+file, 'rb')))
        r = requests.post(
            'http://localhost:6666/compare_signature/', files=files)
        data = r.json()
        if('similarity' in data):
            csv_writer.writerow([file, data['status'], data['similarity']])
        else:
            csv_writer.writerow([file, data['status'], '-1'])
