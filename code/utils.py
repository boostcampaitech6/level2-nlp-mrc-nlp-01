def load_json(f_name):
    import json 
    with open(f_name, 'r') as f:
        file = json.load(f)
    return file
