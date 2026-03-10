import json

with open('/home/shobs/Desktop/DDP/v8.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "sol = odeint_adjoint(" in line:
                # We need to find the if use_adjoint block
                pass
            if "use_adjoint=False" in line and "initial_brain_params['alpha']" in source[i-2]:
                source[i] = line.replace("use_adjoint=False", "use_adjoint=True")
                
        # To replace TorchRevHopfNetwork.solve's odeint_adjoint call
        s_text = "".join(source)
        if "def solve(" in s_text:
            s_text = s_text.replace(
                "        if use_adjoint:\n            sol = odeint_adjoint(\n                self.ode_func,\n                y0,\n                t_eval_tensor,\n                method=\"rk4\"\n            )\n",
                "        if use_adjoint:\n            adjoint_params = tuple(self.ode_func.parameters())\n            if hasattr(self.ode_func, 'brain_drive_full') and self.ode_func.brain_drive_full is not None and self.ode_func.brain_drive_full.requires_grad:\n                adjoint_params = adjoint_params + (self.ode_func.brain_drive_full,)\n            sol = odeint_adjoint(\n                self.ode_func,\n                y0,\n                t_eval_tensor,\n                method=\"rk4\",\n                adjoint_params=adjoint_params\n            )\n"
            )
            # Re-split into list of lines, preserving the \n
            new_source = []
            lines = s_text.splitlines(keepends=True)
            cell['source'] = lines

with open('/home/shobs/Desktop/DDP/v8.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

