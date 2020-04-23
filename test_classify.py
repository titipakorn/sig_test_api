import requests
import os
import csv
path = '/data/CIM/Test_API/'

with open('check_signature.csv', mode='w') as compare_result_file:
    csv_writer = csv.writer(
        compare_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for file in filter(os.listdir(path+'check/'), '*.[Pp][Dd][Ff]'):
        files = []
        files.append(('file', open(path+'check/'+file, 'rb')))
        r = requests.post(
            'http://localhost:6666/check_signature/', files=files)
        data = r.json()
        csv_writer.writerow([file, data.status])
