import os

def find_non_ascii(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            try:
                                line.encode('ascii')
                            except UnicodeEncodeError:
                                print(f"{filepath}:{i+1}: {line.strip()}")
                except Exception as e:
                    pass

find_non_ascii(r'E:\PIMA\PIMA_code')
