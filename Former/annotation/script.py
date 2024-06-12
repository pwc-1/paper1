from glob import glob
import os

your_dataset_path = "/home/s5u1/mafuyan/Dataset/DFEW/images/"
all_txt_file = glob(os.path.join('set*.txt'))


def update(file, old_str, new_str):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str,new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


for txt_file in all_txt_file:
    update(txt_file, "/home/hnu/Pictures/DFEW/train_test/clip224/", your_dataset_path)
