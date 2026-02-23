import json

with open('/home/shobs/Desktop/DDP/shobhith_code_v2.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        
        # We need to change the slice from 2000:4000 to 2000:17000
        if '2000:4000' in source:
            new_source = []
            for line in cell.get('source', []):
                if '2000:4000' in line:
                    new_source.append(line.replace('2000:4000', '2000:17000'))
                else:
                    new_source.append(line)
            cell['source'] = new_source

        if 'def train_heart_model' in source:
            new_source = []
            for line in cell.get('source', []):
                if 'simulate_coupled_oscillators(T=t_duration' in line:
                    # just to be super safe about scope, replace with T=15
                    new_source.append(line.replace('T=t_duration', 'T=15'))
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
with open('/home/shobs/Desktop/DDP/shobhith_code_v2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
