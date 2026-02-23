import json
import re

notebook_path = "/home/shobs/Desktop/DDP/shobhith_code_v2.ipynb"

def fix_notebook():
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    cells = nb['cells']
    modified = False

    for cell in cells:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "class ODEFuc" in source:
                print("Found ODEFuc cell.")
                
                # 1. Fix Phase Difference
                old_phase = r"""    phase_diff = \(
        phi\[None, :\] / omega_safe\[None, :\]
        - phi\[:, None\] / omega_safe\[:, None\]
        \+ theta / \(omega_safe\[:, None\] \* omega_safe\[None, :\]\)
    \)"""
                new_phase = """    # --- Correct Phase Difference for Power Coupling ---
        phase_diff = (
            omega_ratio * phi[None, :]
            - phi[:, None]
            + theta
        )"""
                # Use string replace instead of regex for safety if exact match
                # Constructing exact string from what we saw in view_file
                
                # Let's operate on the source list directly to be safe with line breaks
                new_source = []
                # Helper to replace a block of lines
                
                # Strategy: We will rewrite the content of this specific cell completely
                # based on the known structure, or use regex substitution on the full joined string.
                
                # Regex is risky with indentation. Let's try simple string replacement on the joined source.
                
                # 1. Phase Diff
                target_phase = """        phase_diff = (
            phi[None, :] / omega_safe[None, :]
            - phi[:, None] / omega_safe[:, None]
            + theta / (omega_safe[:, None] * omega_safe[None, :])
        )"""
                
                replacement_phase = """        # --- Correct Phase Difference for Power Coupling ---
        phase_diff = (
            omega_ratio * phi[None, :]
            - phi[:, None]
            + theta
        )"""
                
                if target_phase in source:
                    source = source.replace(target_phase, replacement_phase)
                    print("Fixed Phase Difference.")
                else:
                    print("WARNING: Phase Difference target not found exactly.")

                # 2. Coupling R and Log Domain and Drdt
                target_coupling = """        # ----- Dynamics -----
        coupling_r = torch.sum(
            torch.abs(self.Sc) *
            r[None, :] **omega_ratio*
            torch.cos(phase_diff),
            dim=1
        )

        drdt = (self.mu - r**2) * r \\
               + coupling_r \\
               + e * torch.cos(phi) \\
               + ecg_input"""
               
                replacement_coupling = """        # ----- Dynamics -----
        # Safe Log-Domain Power
        log_r = torch.log(r_safe)
        r_power = torch.exp(omega_ratio * log_r[None, :])

        coupling_r = torch.sum(
            torch.abs(self.Sc)
            * r_power
            * torch.cos(phase_diff),
            dim=1
        )

        drdt = (self.mu - r**2) * r \\
               + coupling_r \\
               + e * torch.cos(phi) \\
               + ecg_input * torch.cos(phi)"""
               
                if target_coupling in source:
                    source = source.replace(target_coupling, replacement_coupling)
                    print("Fixed Coupling R and Drdt.")
                else:
                    print("WARNING: Coupling R target not found exactly.")

                # 3. Coupling Phi
                target_coupling_phi = """        coupling_phi = torch.sum(
            torch.abs(self.Sc) *
            ((r[None, :]**omega_ratio) / r_safe[:, None]) *
            torch.sin(phase_diff),
            dim=1
        )"""
        
                replacement_coupling_phi = """        coupling_phi = torch.sum(
            torch.abs(self.Sc)
            * (r_power / r_safe[:, None])
            * torch.sin(phase_diff),
            dim=1
        )"""
                
                if target_coupling_phi in source:
                    source = source.replace(target_coupling_phi, replacement_coupling_phi)
                    print("Fixed Coupling Phi.")
                else:
                    print("WARNING: Coupling Phi target not found exactly.")

                # 4. Duplicate Class
                # Check if TorchRevHopfNetwork definition appears twice
                class_def = "class TorchRevHopfNetwork:"
                first_idx = source.find(class_def)
                second_idx = source.find(class_def, first_idx + 1)
                
                if second_idx != -1:
                    print("Found duplicate TorchRevHopfNetwork. Removing the second one.")
                    # Keep everything up to the second occurrence
                    # But we need to see what comes after.
                    # The cell ends with return ... rcos_phi
                    
                    # Assuming the second definition goes to the end of the cell or close to it.
                    # Let's look at the structure.
                    # It goes:
                    # class ODEFuc ...
                    # class TorchRevHopfNetwork ...
                    # class TorchRevHopfNetwork ...
                    
                    # We can remove the chunk from first_idx to second_idx (removing the FIRST one)
                    # OR remove from second_idx to end.
                    # Let's remove the second one.
                    # The second one starts at second_idx.
                    # Does it act as a replacement? Yes.
                    
                    # Let's verify if they are identical.
                    # Extract duplicate check is hard string-wise.
                    # Let's simply truncate the source before the second definition if it looks like a full redefinition.
                    
                    # Correction: modifying the source to remove the DUPLICATE.
                    # Since we are in the cell, and we know the last part of the cell is the duplicate.
                    # The cell's last line is `return r, phi, theta, omega, alpha, rcos_phi`
                    # The duplicate starts at `class TorchRevHopfNetwork:`
                    
                    # Let's remove the text from second_idx to the end, BUT wait,
                    # is there anything AFTER the class definition in that cell?
                    # `view_file` showed the cell ending with the class method `solve`.
                    
                    # Actually, if I remove the second one, I keep the first one.
                    # Is the first one modified by my previous replacements? No, ODEFuc is independent.
                    # Wait, TorchRevHopfNetwork uses ODEFuc.
                    # If I keep the first one, it will use the (modified) ODEFuc class because ODEFuc is defined before it in the file scope.
                    
                    # Let's just remove the second occurrence text block.
                    # We have to be careful about matching the *entire* block.
                    # Simpler strategy: The ODEFuc class is fixed in place.
                    # The double definition is just waste.
                    # Let's look for the string `class TorchRevHopfNetwork:`
                    # and split.
                    
                    parts = source.split(class_def)
                    if len(parts) > 2:
                        # part[0] is ODEFuc code
                        # part[1] is body of first TorchRevHopfNetwork
                        # part[2] is body of second TorchRevHopfNetwork
                        
                        # We want part[0] + class_def + part[1] (or part[2] if they differ).
                        # Let's assuming they are identical and keep the first one.
                        
                        # Wait, `split` removes the separator.
                        # New source = part[0] + class_def + part[1]
                        
                        # We need to trim part[1] if it has trailing stuff or if part[2] has leading stuff.
                        # The split happens exactly at "class ...".
                        
                        source = parts[0] + class_def + parts[1]
                        print("Removed duplicate class definition.")
                    
                
                # Split back into lines for JSON
                # We need to keep the \n at end of lines if they were there
                # "source" is a list of strings including \n. 
                # "".join puts them together.
                # When splitting we need to respect lines.
                
                # Re-split by lines and add \n (except maybe last one)
                new_lines = source.splitlines(keepends=True)
                cell['source'] = new_lines
                modified = True
                
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1) # using indent 1 to be concise but readable
        print("Notebook saved successfully.")
    else:
        print("No changes made. Targets possibly not found.")

if __name__ == "__main__":
    fix_notebook()
