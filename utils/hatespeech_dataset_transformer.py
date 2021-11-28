file = open("datasets/hate-speech/annotations_metadata.csv", "r")
write_file = open("datasets/hate-speech/data.txt", "w+")
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
    data_file = open(f'datasets/hate-speech/all_files/{file_name}.txt', "r")
    content = data_file.read()
    data_file.close()
    write_file.write("\t".join([content, label]) + "\n")
    count += 1
    print(count)
