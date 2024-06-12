def read_and_merge_files(file_names):
    """
    读取多个文件，合并内容，并去除重复项。
    :param file_names: 文件名列表
    :return: 去重后的数据列表
    """
    data_set = set()  # 使用集合去重
    for file_name in file_names:
        with open(file_name, 'r') as file:
            for line in file:
                data_set.add(line.strip())  # 去除尾部的换行符，并添加到集合中去重
    return list(data_set)

def write_to_file(file_name, data_list):
    """
    将数据写入文件。
    :param file_name: 文件名
    :param data_list: 数据列表
    """
    with open(file_name, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")  # 写入数据并添加换行符

# 文件名
file_names = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt']

# 读取和合并文件
merged_data = read_and_merge_files(file_names)

# 输出到新文件
write_to_file('trainlist.txt', merged_data)

print("数据合并完成，输出到trainlist.txt")
