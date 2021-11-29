from math import ceil
import os
file = open("datasets/hate-speech/annotations_metadata.csv", "r")
list_of_files = [file_name for file_name in os.listdir(
    "datasets/hate-speech/sampled_test") if file_name.endswith(".txt")]
write_file = open("datasets/hate-speech/test.txt", "w+")
data = file.read().split()
file.close()
count = 0
for line in data:
    print(line)
    words = line.split(",")
    file_name = words[0]
    if file_name == "file_id":
        continue
    label = words[-1]
    file_name_ext = file_name+".txt"
    if file_name_ext not in list_of_files:
        continue
    data_file = open(
        f'datasets/hate-speech/sampled_test/{file_name}.txt', "r")
    content = data_file.read()
    data_file.close()
    write_file.write("\t".join([content, label]) + "\n")
    count += 1
    print(count)
