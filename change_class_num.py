import os

def update_label_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                with open(file_path, 'w') as f:
                    for line in lines:
                        # Replace the first occurrence of '0' with '80'
                        line = line.replace('80 ', '0 ', 1)
                        f.write(line)

# Update label files in train folder
update_label_files('train')

# Update label files in test folder
update_label_files('test')

# Update label files in val folder
update_label_files('val')
