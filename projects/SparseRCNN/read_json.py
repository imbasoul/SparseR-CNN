import json
from tqdm import tqdm
with open("/data/output/inference/coco_instances_results.json") as f:
    load_dict = json.load(f)
    for instance in tqdm(load_dict):
        file_name = instance["image_id"][:-4] + ".txt"
        with open("/data/animation_face/submission-new-res101-144999/" + file_name, "a") as write_file:
            if instance["score"] > 0.05:
                write_file.write("face " + str(instance["score"]) + " " + str(instance["bbox"][0]) + " " + str(instance["bbox"][1]) + " " + str(instance["bbox"][0]+instance["bbox"][2]) + " " + str(instance["bbox"][1]+instance["bbox"][3]) + "\n")