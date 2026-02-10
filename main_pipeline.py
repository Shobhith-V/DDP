"""
Entry point that wires together data loading, preprocessing, model
training and feedback simulation.

This is a modular version of the logic that originally lived in
`Shobhith_sent.ipynb`.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    DEFAULT_ECG_FIF_PATH,
    DEFAULT_EEG_SCOUT_MAT_PATH,
    DEFAULT_SC_MAT_PATH,
)
from data_loading import load_ecg_eeg_and_connectivity
from preprocessing import preprocess_ecg_eeg
from training import (
    train_heart_model,
    pre_train_brain_model,
    train_mlp_on_frozen_brain,
    train_feedback_loop,
    make_target_function_from_eeg,
)
from models_brain_ode import TorchRevHopfNetwork
from oscillator_utils import simulate_coupled_oscillators


def run_full_pipeline(
    ecg_fif_path: str = DEFAULT_ECG_FIF_PATH,
    eeg_mat_path: str = DEFAULT_EEG_SCOUT_MAT_PATH,
    sc_mat_path: str = DEFAULT_SC_MAT_PATH,
) -> None:
    # --- LOAD & PREPROCESS DATA ---
    ecg_data, eeg_data, Sw_all, non_zero_indices_per_row = (
        load_ecg_eeg_and_connectivity(ecg_fif_path, eeg_mat_path, sc_mat_path)
    )
    ecg_processed, eeg_processed = preprocess_ecg_eeg(ecg_data, eeg_data)

    target_indices = [46]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    # Step 1: Pre-train heart
    trained_heart_model = train_heart_model(ecg_processed, device)

    results_folder = Path("simulation_results")
    results_folder.mkdir(exist_ok=True)

    for target_idx in target_indices:
        t_duration = 2.0
        fs = 100.0
        t = np.arange(0, t_duration, 1 / fs)

        # Step 2: Stage 1 - Brain pre-training
        target_signal, D_function = make_target_function_from_eeg(
            eeg_processed, target_idx, t
        )
        final_brain_params, Sc_reduced_osc, N, brain_losses = pre_train_brain_model(
            eeg_processed,
            Sw_all,
            target_idx,
            non_zero_indices_per_row,
            t,
            D_function,
            device,
        )

        # Step 3: Stage 2 - MLP training
        trained_mlp_model, mlp_losses = train_mlp_on_frozen_brain(
            trained_heart_model,
            final_brain_params,
            Sc_reduced_osc,
            N,
            D_function,
            t,
            device,
        )

        # Step 4: Extract rcos_phi from final brain params
        print("\n--- Extracting Brain rcos_phi for Feedback ---")
        model_final = TorchRevHopfNetwork(
            mu=1.0,
            eta_omega=0.0,
            eta_alpha=0.0,
            eta_theta=0.0,
            D_function=D_function,
            N=N,
            Sc=Sc_reduced_osc,
            mlp_model=None,
            hidden_repr=None,
            device=device,
        )
        r_final, phi_final, theta_final, omega_final, alpha_final, rcos_phi_final = (
            model_final.solve(
                final_brain_params["r"],
                final_brain_params["phi"],
                final_brain_params["theta"],
                final_brain_params["omega"],
                final_brain_params["alpha"],
                t,
            )
        )
        print(
            f"rcos_phi_final shape: {rcos_phi_final.shape}, "
            f"range: [{rcos_phi_final.min():.3f}, {rcos_phi_final.max():.3f}]"
        )

        # Step 5: Feedback training with brain rcos_phi
        trained_heart_model, trained_feedback_mlp, feedback_losses = (
            train_feedback_loop(
                trained_heart_model,
                rcos_phi_final.cpu().numpy(),
                ecg_processed,
                T=t_duration,
                dt=1 / fs,
                device=device,
                num_epochs=5000,
            )
        )

        # FINAL PREDICTION & PLOTTING
        print("\n--- Final Predictions ---")
        trained_heart_model.eval()
        trained_feedback_mlp.eval()

        with torch.no_grad():
            # Baseline ECG (no feedback)
            sim_osc_baseline = simulate_coupled_oscillators(
                T=t_duration, dt=1 / fs
            )
            predicted_ecg_baseline = (
                trained_heart_model(
                    torch.tensor(
                        sim_osc_baseline, dtype=torch.float32
                    ).to(device)
                )
                .cpu()
                .numpy()
                .flatten()
            )

            # Feedback ECG
            feedback_output = trained_feedback_mlp(
                torch.tensor(
                    rcos_phi_final.cpu().numpy(), dtype=torch.float32
                ).to(device)
            )
            modulation = feedback_output.mean(dim=0).cpu().numpy()
            sim_osc_feedback = simulate_coupled_oscillators(
                T=t_duration,
                dt=1 / fs,
                modulation=np.tile(modulation, (len(t), 1)),
            )
            predicted_ecg_feedback = (
                trained_heart_model(
                    torch.tensor(
                        sim_osc_feedback, dtype=torch.float32
                    ).to(device)
                )
                .cpu()
                .numpy()
                .flatten()
            )

            # Baseline EEG brain output
            P_out_baseline = torch.sum(
                alpha_final * r_final * torch.cos(phi_final), axis=1
            ).cpu().numpy()

        # PLOTTING
        fig, axes = plt.subplots(5, 1, figsize=(15, 20))

        axes[0].plot(brain_losses)
        axes[0].set_title("Stage 1: Brain Pre-training Loss")
        axes[0].grid(True)

        axes[1].plot(mlp_losses)
        axes[1].set_title("Stage 2: MLP Training Loss")
        axes[1].grid(True)

        axes[2].plot(feedback_losses)
        axes[2].set_title("Stage 3: Brain→Feedback→Heart Loss")
        axes[2].grid(True)

        target_ecg = ecg_processed[::10]
        timesteps = np.linspace(0, t_duration, len(target_ecg))

        axes[3].plot(timesteps, target_ecg, label="Target ECG", linewidth=2)
        axes[3].plot(
            timesteps,
            predicted_ecg_baseline,
            label="Baseline ECG",
            linestyle="--",
        )
        axes[3].plot(
            timesteps,
            predicted_ecg_feedback,
            label="Feedback ECG",
            linestyle=":",
        )
        axes[3].set_title("ECG Prediction: Baseline vs Brain Feedback")
        axes[3].legend()
        axes[3].grid(True)

        axes[4].plot(t, D_function(t), label="Target EEG", linewidth=2)
        axes[4].plot(
            t,
            P_out_baseline,
            label="P_out baseline",
            alpha=0.7,
        )
        axes[4].set_title("Brain Output: rcos_phi vs Target")
        axes[4].legend()
        axes[4].grid(True)

        plt.tight_layout()
        out_png = results_folder / f"full_feedback_result_idx{target_idx}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.show()

        # Save results arrays
        out_npz = results_folder / f"results_idx{target_idx}.npz"
        np.savez(
            out_npz,
            brain_losses=brain_losses,
            mlp_losses=mlp_losses,
            feedback_losses=feedback_losses,
            rcos_phi_final=rcos_phi_final.cpu().numpy(),
            P_out_baseline=P_out_baseline,
            predicted_ecg_baseline=predicted_ecg_baseline,
            predicted_ecg_feedback=predicted_ecg_feedback,
            target_ecg=target_ecg,
            target_eeg=D_function(t),
        )

    print("✅ COMPLETE! Check simulation_results/ folder")


if __name__ == "__main__":
    run_full_pipeline()

