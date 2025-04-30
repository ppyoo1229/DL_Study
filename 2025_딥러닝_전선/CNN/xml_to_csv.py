import os
import xml.etree.ElementTree as ET
import pandas as pd

CLASS_LABELS = ["Face Mask", "Gloves", "Helmet", "No Gloves", "No Helmet", "No Mask"]

def label_vector(obj_names, class_list):
    vec = [0] * len(class_list)
    for name in obj_names:
        if name in class_list:
            vec[class_list.index(name)] = 1
    return vec

def xml_to_csv(xml_folder, image_folder, output_csv):
    rows = []
    for file in os.listdir(xml_folder):
        if not file.endswith(".xml"):
            continue
        path = os.path.join(xml_folder, file)
        tree = ET.parse(path)
        root = tree.getroot()

        filename = root.find('filename').text
        obj_names = [obj.find('name').text for obj in root.findall('object')]
        label_vec = label_vector(obj_names, CLASS_LABELS)

        rows.append([os.path.join(image_folder, filename)] + label_vec)

    df = pd.DataFrame(rows, columns=["filename"] + CLASS_LABELS)
    df.to_csv(output_csv, index=False)
    print(f"{output_csv} 저장")

# 사용 예
xml_to_csv("train", "train", "train_labels_multilabel.csv")

