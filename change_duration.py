import json

with open('/home/shobs/Desktop/DDP/shobhith_code_v2.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        
        # Update t_duration
        if 't_duration = 2' in source:
            new_source = []
            for line in cell['source']:
                if 't_duration = 2\n' in line:
                    new_source.append(line.replace('t_duration = 2', 't_duration = 15'))
                elif 't_duration = 2' in line:
                    new_source.append(line.replace('t_duration = 2', 't_duration = 15'))
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
            # Since sim_osc_baseline in the final prediction is hardcoded to T=t_duration, that is fine.
            
        # Update the MLP feature extraction which is hardcoded (T=2)
        if 'simulate_coupled_oscillators(T=2' in source:
            new_source = []
            for line in cell['source']:
                if 'simulate_coupled_oscillators(T=2' in line:
                    new_source.append(line.replace('T=2', 'T=t_duration'))
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
        # Update split_idx in analysis cell
        if 'split_idx = int(0.8 * len(t))' in source:
            new_source = []
            for line in cell['source']:
                if 'split_idx = int(0.8 * len(t))' in line:
                    # 10s train, 5s test out of 15s total means 10/15 = 2/3 split
                    new_source.append("split_idx = int((10.0 / 15.0) * len(t))\n")
                elif 'We split the sequence into Train (80%) and Test (20%)' in line:
                    new_source.append(line.replace('Train (80%) and Test (20%)', 'Train (10s) and Test (5s)'))
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    if cell.get('cell_type') == 'markdown':
        source = "".join(cell.get('source', []))
        if 'We split the sequence into Train (80%) and Test (20%)' in source:
             new_source = []
             for line in cell['source']:
                 if 'We split the sequence into Train (80%) and Test (20%)' in line:
                     new_source.append(line.replace('Train (80%) and Test (20%)', 'Train (10s) and Test (5s)'))
                 else:
                     new_source.append(line)
             cell['source'] = new_source

with open('/home/shobs/Desktop/DDP/shobhith_code_v2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
