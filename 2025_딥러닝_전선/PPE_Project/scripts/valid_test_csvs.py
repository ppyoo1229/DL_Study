import os
import xml.etree.ElementTree as ET
import pandas as pd

CLASS_LABELS = ["Face Mask", "Gloves", "Helmet", "No Gloves", "No Helmet", "No Mask"]

def make_label_vector(obj_names, class_list):
    vec = [0] * len(class_list)
    for name in obj_names:
        if name in class_list:
            vec[class_list.index(name)] = 1
    return vec

def xml_to_multilabel_csv(xml_folder, img_folder, output_csv):
    rows = []
    for f in os.listdir(xml_folder):
        if not f.endswith(".xml"):
            continue
        root = ET.parse(os.path.join(xml_folder, f)).getroot()
        fn = root.find("filename").text
        objs = [o.find("name").text for o in root.findall("object")]
        vec = make_label_vector(objs, CLASS_LABELS)
        img_path = os.path.join(img_folder, fn)
        rows.append([img_path] + vec)

    df = pd.DataFrame(rows, columns=["filename"] + CLASS_LABELS)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("Saved:", output_csv)

if __name__ == "__main__":
    BASE_RAW = r"C:\Users\sj123\Doit_DeepLearning\2025_딥러닝_전선\PPE_Data\raw"

    # valid
    xml_to_multilabel_csv(
        xml_folder=os.path.join(BASE_RAW, "valid"),
        img_folder=os.path.join(BASE_RAW, "valid"),
        output_csv=r"C:\Users\sj123\Doit_DeepLearning\2025_딥러닝_전선\PPE_Data\processed\valid_multilabels.csv"
    )

    # test
    xml_to_multilabel_csv(
        xml_folder=os.path.join(BASE_RAW, "test"),
        img_folder=os.path.join(BASE_RAW, "test"),
        output_csv=r"C:\Users\sj123\Doit_DeepLearning\2025_딥러닝_전선\PPE_Data\processed\test_multilabels.csv"
    )
