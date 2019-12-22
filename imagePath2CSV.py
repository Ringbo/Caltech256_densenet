import os
import sys
import csv

IMAGE_PATH = os.path.join(sys.path[0], 'data', 'input')
f = open('dev.csv', 'w', encoding='utf-8', newline='')
csvWriter = csv.writer(f)

# header
csvWriter.writerow(['image_path	label', 'label'])

for root, dirs, files in os.walk(IMAGE_PATH):
    for file in files:
        lable = file.split('_')[0]
        csvWriter.writerow([os.path.join(root.replace(IMAGE_PATH, '', 1), file), int(lable) - 1])
