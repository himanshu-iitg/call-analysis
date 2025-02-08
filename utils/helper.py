# We will keep all the util functions here
import json
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
