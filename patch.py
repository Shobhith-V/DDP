import re

with open('/home/shobs/Desktop/DDP/working.py', 'r') as f:
    text = f.read()

# 1. OscillatorLayer
osc_regex = r"class OscillatorLayer\(nn\.Module\):.*?(?=class ECGToOscillatorMLP\(nn\.Module\):)"
new_osc = '''def compute_power_terms(r, phi, omega, theta):
    omega_safe = torch.clamp(omega, 2 * torch.pi * 0.5, 2 * torch.pi * 20.0)
    rho = omega_safe.unsqueeze(-1) / omega_safe.unsqueeze(-2)
    rho = torch.clamp(rho, 0.1, 10.0)

    phase_diff = rho * phi.unsqueeze(-2) - phi.unsqueeze(-1) + theta
    phase_diff = torch.clamp(phase_diff, -50.0, 50.0)

    r_safe = torch.clamp(r, 1e-5, 10.0)
    log_r = torch.log(r_safe)
    r_power = torch.exp(rho * log_r.unsqueeze(-2))
    r_power = torch.clamp(r_power, 1e-6, 10.0)

    return phase_diff, r_power, r_safe, rho

class OscillatorLayer(nn.Module):
    def __init__(self, N_osc=16, T=2.0, fs=100, device="cpu",
                 coupling_sparsity=0.3, seed=42):

        super().__init__()

        self.N_osc = N_osc
        self.original_dt = 1.0 / fs
        self.substeps = 4
        self.dt = self.original_dt / self.substeps
        self.num_steps = int(T * fs) * self.substeps
        self.mu = 1.0
        self.k = 0.01

        torch.manual_seed(seed)

        freqs = 2.0 + torch.rand(N_osc, device=device) * 8.0
        self.omega = nn.Parameter(2 * torch.pi * freqs)

        self.register_buffer("initial_r", torch.ones(N_osc, device=device) * 0.1)
        self.register_buffer("initial_phi", torch.zeros(N_osc, device=device))

        mask = torch.rand(N_osc, N_osc, device=device) > coupling_sparsity
        mask.fill_diagonal_(False)

        C = torch.rand(N_osc, N_osc, device=device) * 0.02
        self.register_buffer("C", C * mask.float())

        theta = torch.zeros(N_osc, N_osc, device=device)
        theta.fill_diagonal_(0.0)
        self.register_buffer("theta", theta)

        # --- Stability clamps ---
        self.r_min = 0.01
        self.r_max = 2.0

    def forward(self, input_features):
        if input_features.dim() == 1:
            input_features = input_features.unsqueeze(0)

        B = input_features.shape[0]

        r = self.initial_r.unsqueeze(0).repeat(B, 1)
        phi = self.initial_phi.unsqueeze(0).repeat(B, 1)

        input_f = input_features

        omega = self.omega
        C = self.C
        theta = self.theta
        mu_t = self.mu

        for _ in range(self.num_steps):
            phase_diff, r_power, r_safe, rho = compute_power_terms(r, phi, omega, theta)

            coupling_r = self.k * torch.sum(
                C * r_power * torch.cos(phase_diff),
                dim=-1
            )

            coupling_phi = self.k * torch.sum(
                C * (r_power / r_safe.unsqueeze(-1)) * torch.sin(phase_diff),
                dim=-1
            )

            Fx = input_f
            Fy = torch.zeros_like(Fx)

            dr_dt = (mu_t - r**2) * r + coupling_r + Fx * torch.cos(phi) + Fy * torch.sin(phi)
            dphi_dt = omega + coupling_phi + (-Fx * torch.sin(phi) + Fy * torch.cos(phi)) / r_safe

            r = r + dr_dt * self.dt
            phi = phi + dphi_dt * self.dt

            r = torch.clamp(r, self.r_min, self.r_max)

        return torch.cat([r * torch.cos(phi), r * torch.sin(phi)], dim=-1)

'''
text = re.sub(osc_regex, new_osc, text, flags=re.DOTALL)

# 2. ECGToOscillatorMLP Section 9
old_ecg_mlp = '''        osc_hidden = self.osc_layer(pre)         # (B, N_VNS * 2)'''
new_ecg_mlp = '''        osc_hidden = self.osc_layer(pre) + torch.cat([pre, pre], dim=-1)'''
text = text.replace(old_ecg_mlp, new_ecg_mlp)

# 3. ODEFuc
odefuc_regex = r"class ODEFuc\(nn\.Module\):.*?(?=class TorchRevHopfNetwork:)"
new_odefuc = '''class ODEFuc(nn.Module):
    def __init__(self, mu, eta_theta, eta_omega, eta_alpha,
                 D_function, N, Sc,
                 brain_drive_full=None,
                 fs=100):

        super().__init__()
        self.mu = mu
        self.eta_theta = eta_theta
        self.eta_omega = eta_omega
        self.eta_alpha = eta_alpha
        self.D_function = D_function
        self.N = N
        self.fs = fs

        self.register_buffer(
            'Sc',
            torch.tensor(Sc, dtype=torch.float32)
        )
        
        self.readout = nn.Linear(N, 1, bias=False)

        self.brain_drive_full = brain_drive_full

    def forward(self, t, state):
        N = self.N

        r = state[:N]
        phi = state[N:2*N]
        theta = state[2*N:2*N + N**2].view(N, N)
        omega = state[2*N + N**2:3*N + N**2]
        alpha = state[3*N + N**2:4*N + N**2]

        r = torch.clamp(r, 1e-2, 2.0)
        alpha = torch.clamp(alpha, -1.0, 1.0)

        phase_diff, r_power, r_safe, rho = compute_power_terms(r, phi, omega, theta)

        D = torch.tensor(
            self.D_function(t.item()),
            device=state.device,
            dtype=state.dtype
        )

        brain_state = r * torch.cos(phi)
        P = self.readout(brain_state).squeeze(-1)
        e = D - P

        if self.brain_drive_full is not None:
            t_idx = min(
                int(t.item() * self.fs),
                self.brain_drive_full.shape[0] - 1
            )
            ecg_input = self.brain_drive_full[t_idx]
        else:
            ecg_input = torch.zeros(N, device=state.device)

        coupling_r = torch.sum(
            self.Sc
            * r_power
            * torch.cos(phase_diff),
            dim=1
        )

        Fx_e = e
        Fy_e = torch.zeros_like(Fx_e)

        Fx_ecg = ecg_input
        Fy_ecg = torch.zeros_like(Fx_ecg)

        drdt = (self.mu - r**2) * r \\
               + coupling_r \\
               + Fx_e * torch.cos(phi) + Fy_e * torch.sin(phi) \\
               + Fx_ecg * torch.cos(phi) + Fy_ecg * torch.sin(phi)

        coupling_phi = torch.sum(
            self.Sc
            * (r_power / r_safe.unsqueeze(-1))
            * torch.sin(phase_diff),
            dim=1
        )

        dphidt = omega + coupling_phi \\
                 + (-Fx_e * torch.sin(phi) + Fy_e * torch.cos(phi)) / r_safe \\
                 + (-Fx_ecg * torch.sin(phi) + Fy_ecg * torch.cos(phi)) / r_safe

        dthetadt = self.eta_theta * torch.sin(phase_diff) * self.Sc
        domegadt = -self.eta_omega * e * torch.sin(phi)
        dalphadt = self.eta_alpha * e * r * torch.cos(phi)

        return torch.cat([
            drdt.flatten(),
            dphidt.flatten(),
            dthetadt.flatten(),
            domegadt.flatten(),
            dalphadt.flatten()
        ])

'''
text = re.sub(odefuc_regex, new_odefuc, text, flags=re.DOTALL)

# 4. Feature Normalization
old_feat = '''        hidden_repr = trained_heart_model.get_features(sim_input)  # (T_steps, feature_dim)'''
new_feat = '''        hidden_repr = trained_heart_model.get_features(sim_input)  # (T_steps, feature_dim)
        hidden_repr = (hidden_repr - hidden_repr.mean(dim=0)) / (hidden_repr.std(dim=0) + 1e-6)'''
text = text.replace(old_feat, new_feat)

# 5. Optimizer Update for MLP
old_opt = '''    optimizer = torch.optim.Adam(
        ecg_to_osc_mlp.parameters(),
        lr=5e-3
    )'''
new_opt = '''    optimizer = torch.optim.AdamW(
        ecg_to_osc_mlp.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.99)
    )'''
text = text.replace(old_opt, new_opt)

# Gradient clipping update
old_clip = '''        torch.nn.utils.clip_grad_norm_(ecg_to_osc_mlp.parameters(), max_norm=1.0)'''
new_clip = '''        torch.nn.utils.clip_grad_norm_(ecg_to_osc_mlp.parameters(), 0.3)'''
text = text.replace(old_clip, new_clip)

# 6,7,8. Normalized brain drive, temporal smoothing, curriculum
old_drive = '''        raw_drive = ecg_to_osc_mlp(hidden_repr)  # (T_steps, N) — has grad_fn

        brain_drive_full = torch.tanh(raw_drive)'''
new_drive = '''        raw_drive = ecg_to_osc_mlp(hidden_repr)  # (T_steps, N) — has grad_fn

        drive = raw_drive
        drive = drive / (drive.std(dim=0, keepdim=True) + 1e-6)
        brain_drive_full = 0.05 * drive

        brain_drive_full = torch.nn.functional.avg_pool1d(
            brain_drive_full.T.unsqueeze(0),
            kernel_size=5,
            stride=1,
            padding=2
        ).squeeze(0).T

        scale = min(1.0, epoch / 50.0)
        brain_drive_full = brain_drive_full * scale'''
text = text.replace(old_drive, new_drive)

# 9. Update P_out inside pre_train_brain_model
old_pout1 = '''            P_out = torch.sum(alpha * r * torch.cos(phi), axis=1)'''
new_pout1 = '''            P_out = model.ode_func.readout(r * torch.cos(phi)).squeeze(-1)'''
text = text.replace(old_pout1, new_pout1)

# 10. Update P_out inside train_mlp_on_frozen_brain
old_pout2 = '''        P_out = torch.sum(alpha * r * torch.cos(phi), dim=1)'''
new_pout2 = '''        P_out = model.ode_func.readout(r * torch.cos(phi)).squeeze(-1)'''
text = text.replace(old_pout2, new_pout2)

# 11. Final blocks update
old_fb1 = '''    brain_drive_for_final = torch.tanh(trained_mlp_model(hidden_repr))  ## torch.tanh is not needed
    brain_drive_full = torch.tanh(trained_mlp_model(hidden_repr))  # [T, N]'''
new_fb1 = '''    raw_drive = trained_mlp_model(hidden_repr)
    drive = raw_drive
    drive = drive / (drive.std(dim=0, keepdim=True) + 1e-6)
    brain_drive_full = 0.05 * drive

    brain_drive_full = torch.nn.functional.avg_pool1d(
        brain_drive_full.T.unsqueeze(0),
        kernel_size=5,
        stride=1,
        padding=2
    ).squeeze(0).T'''
text = text.replace(old_fb1, new_fb1)

old_fb2 = '''    P_out_baseline = torch.sum(alpha_final * r_final * torch.cos(phi_final), axis=1).cpu().numpy()'''
new_fb2 = '''    P_out_baseline = model.ode_func.readout(r_final * torch.cos(phi_final)).squeeze(-1).detach().cpu().numpy()'''
text = text.replace(old_fb2, new_fb2)

# Update both metric blocks
old_pm = '''    P_predicted = torch.sum(alpha_final * r_final * torch.cos(phi_final), dim=1).cpu().numpy()'''
new_pm = '''    P_predicted = model.ode_func.readout(r_final * torch.cos(phi_final)).squeeze(-1).detach().cpu().numpy()'''
text = text.replace(old_pm, new_pm)

with open('/home/shobs/Desktop/DDP/working.py', 'w') as f:
    f.write(text)
print("Done!")
