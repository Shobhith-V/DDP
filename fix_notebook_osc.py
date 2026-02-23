import json

notebook_path = "/home/shobs/Desktop/DDP/shobhith_code_v2.ipynb"

def fix_oscillator_layer():
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    cells = nb['cells']
    modified = False

    for cell in cells:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "class OscillatorLayer" in source and "r_power = torch.pow" in source:
                print("Found OscillatorLayer cell.")
                
                target_power = """            # --- Power amplitude ---
            r_j_clamped = torch.clamp(r_j, self.power_min, self.power_max)

            r_power = torch.pow(r_j_clamped, rho)

            # Clamp power explosion
            r_power = torch.clamp(r_power, self.power_min, self.power_max)"""
            
                replacement_power = """            # --- Power amplitude (Log-Domain Safe) ---
            r_j_clamped = torch.clamp(r_j, self.power_min, self.power_max)
            
            log_r_j = torch.log(r_j_clamped)
            r_power = torch.exp(rho * log_r_j)

            # Clamp power explosion
            r_power = torch.clamp(r_power, self.power_min, self.power_max)"""
                
                if target_power in source:
                    source = source.replace(target_power, replacement_power)
                    print("Fixed OscillatorLayer Power Term.")
                    
                    new_lines = source.splitlines(keepends=True)
                    cell['source'] = new_lines
                    modified = True
                else:
                    print("WARNING: OscillatorLayer Power target not found exactly.")
                    # Try fuzzy match or smaller chunk?
                    # The spacing might vary.
                    
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print("OscillatorLayer fixes saved.")
    else:
        print("No changes made to OscillatorLayer.")

if __name__ == "__main__":
    fix_oscillator_layer()
