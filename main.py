#!/usr/bin/env python3
"""CLI entrypoint for coupled heart-brain pipeline."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import torch

from data.loaders import load_all_data
from preprocessing.signals import preprocess_signal
from dynamics.heart_oscillators import simulate_coupled_oscillators_numpy
from dynamics.heart_oscillators_torch import HeartOscillatorTorch
from dynamics.brain_ode import TorchRevHopfNetwork
from models.heart import HeartModel
from models.feedback import FeedbackMLP
from training.heart import train_heart_model
from training.brain import pre_train_brain_model
from training.ecg_brain import train_mlp_on_frozen_brain
from training.feedback import train_feedback_loop
from evaluation.plots import plot_full_results
from utils.seeding import set_seed
from utils.config import load_config
from utils.checkpoint import save_checkpoint, load_checkpoint


@dataclass
class PipelineState:
    ecg_processed: np.ndarray
    eeg_processed: np.ndarray
    sc_matrix: np.ndarray
    non_zero_indices: list
    heart_model: HeartModel | None = None
    brain_params: dict | None = None
    ecg_to_brain_mlp: object = None
    feedback_mlp: FeedbackMLP | None = None
    Sc_reduced: np.ndarray | None = None
    N: int = 0


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def run_pipeline(cfg: dict, args: argparse.Namespace) -> None:
    proj_root = Path(__file__).resolve().parent
    set_seed(cfg.get("training", {}).get("seed", 42))

    output_dir = Path(cfg.get("paths", {}).get("output_dir", "results"))
    ckpt_dir = Path(cfg.get("paths", {}).get("checkpoints_dir", "checkpoints"))
    figures_dir = Path(cfg.get("paths", {}).get("figures_dir", "figures"))
    logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "logs"))
    output_dir = proj_root / output_dir
    ckpt_dir = proj_root / ckpt_dir
    figures_dir = proj_root / figures_dir
    logs_dir = proj_root / logs_dir

    setup_logging(logs_dir)
    logger = logging.getLogger(__name__)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_cfg = cfg.get("data", {})
    preproc_cfg = cfg.get("preprocessing", {})
    paths_cfg = cfg.get("paths", {})

    ecg_raw, eeg_raw, sc_matrix, non_zero = load_all_data(
        ecg_path=data_cfg.get("ecg_path", "transdef_mf2pt2_rest_raw.fif"),
        eeg_path=data_cfg.get("eeg_path", "scout_id_309.mat"),
        sc_path=data_cfg.get("sc_path", "SC_CC120309-27.mat"),
        ecg_channel=data_cfg.get("ecg_channel", 322),
        t_start=data_cfg.get("t_start", 2000),
        t_end=data_cfg.get("t_end", 4000),
        ecg_negate=preproc_cfg.get("ecg_negate", True),
        base_dir=proj_root,
    )

    ecg_processed = preprocess_signal(
        ecg_raw,
        fs=data_cfg.get("fs_raw", 1000),
        lowcut=preproc_cfg.get("ecg_lowcut", 1.5),
        highcut=preproc_cfg.get("ecg_highcut", 20),
    )
    eeg_processed = np.array([
        preprocess_signal(
            row,
            fs=data_cfg.get("fs_raw", 1000),
            lowcut=preproc_cfg.get("eeg_lowcut", 0.5),
            highcut=preproc_cfg.get("eeg_highcut", 20),
        )
        for row in eeg_raw
    ])

    dyn_heart = cfg.get("dynamics", {}).get("heart", {})
    train_cfg = cfg.get("training", {})
    target_indices = cfg.get("target_indices", [4])

    stage = args.stage
    resume = args.resume

    heart_model = None
    if resume and Path(resume).exists():
        ckpt = load_checkpoint(str(resume), device)
        heart_model = HeartModel(**cfg.get("model", {}).get("heart", {})).to(device)
        heart_model.load_state_dict(ckpt.get("heart_model", ckpt))
        logger.info("Loaded heart model from checkpoint")
    if heart_model is None or stage in ("heart", "all"):
        heart_model = train_heart_model(
            ecg_processed,
            device,
            omega1_hz=dyn_heart.get("omega1_hz", 1.0),
            omega2_hz=dyn_heart.get("omega2_hz", 1.2),
            dt=0.01,
            T=2.0,
            num_epochs=train_cfg.get("heart_epochs", 25000),
            lr=train_cfg.get("heart_lr", 1e-3),
            log_interval=train_cfg.get("log_interval", 2500),
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            {"heart_model": heart_model.state_dict(), "stage": "heart"},
            str(ckpt_dir / "heart.pt"),
        )

    if stage == "heart":
        logger.info("Heart-only stage complete.")
        return

    with torch.no_grad():
        sim_osc = simulate_coupled_oscillators_numpy(
            T=2.0, dt=0.01,
            omega1_hz=dyn_heart.get("omega1_hz", 1.0),
            omega2_hz=dyn_heart.get("omega2_hz", 1.2),
        )
        sim_osc_t = torch.tensor(sim_osc, dtype=torch.float32, device=device)
        hidden_repr = heart_model.get_features(sim_osc_t)

    for target_idx in target_indices:
        t_duration = 2.0
        fs = 100
        t = np.arange(0, t_duration, 1 / fs)
        target_signal = eeg_processed[target_idx, ::10]
        D_function = interp1d(t, target_signal, kind="linear", bounds_error=False, fill_value=0.0)

        run_brain = stage in ("brain", "mlp", "feedback", "all")
        if run_brain:
            brain_params, Sc_reduced, N, brain_losses = pre_train_brain_model(
                eeg_processed,
                sc_matrix,
                non_zero,
                target_idx,
                t,
                target_signal,
                device,
                osc_per_region=cfg.get("dynamics", {}).get("brain", {}).get("osc_per_region", 3),
                eta_omega=cfg.get("dynamics", {}).get("brain", {}).get("eta_omega", 0.05),
                eta_alpha=cfg.get("dynamics", {}).get("brain", {}).get("eta_alpha", 0.005),
                eta_theta=cfg.get("dynamics", {}).get("brain", {}).get("eta_theta", 0.05),
                num_epochs=train_cfg.get("brain_epochs", 30),
                seed=train_cfg.get("seed", 42),
            )
        else:
            from utils.connectivity import expand_structural_connectivity, get_random_frequencies
            connected = np.unique(np.append(non_zero[target_idx], target_idx))
            osc_per_region = cfg.get("dynamics", {}).get("brain", {}).get("osc_per_region", 3)
            N = len(connected) * osc_per_region
            Sc_regional = sc_matrix[np.ix_(connected, connected)]
            Sc_reduced = expand_structural_connectivity(Sc_regional, osc_per_region, seed=train_cfg.get("seed", 42))
            omega_full = get_random_frequencies(68, osc_per_region, 1, 20, train_cfg.get("seed"))
            alpha_full = np.random.uniform(0.1, 0.7, 68 * osc_per_region)
            omega0 = np.concatenate([omega_full[i * osc_per_region : (i + 1) * osc_per_region] for i in connected])
            alpha0 = np.clip(np.concatenate([alpha_full[i * osc_per_region : (i + 1) * osc_per_region] for i in connected]), 0.05, 0.5)
            theta0 = np.pi * (2 * np.random.rand(N, N) - 1)
            theta0 = theta0 - theta0.T
            brain_params = {"r": 0.1 * np.ones(N), "phi": np.zeros(N), "theta": theta0, "omega": omega0, "alpha": alpha0}
            brain_losses = []

        D_tensor = torch.tensor(target_signal, dtype=torch.float32, device=device)
        t_eval = torch.tensor(t, dtype=torch.float32, device=device)

        model_cfg = cfg.get("model", {})
        ecg_cfg = model_cfg.get("ecg_to_brain", {})
        osc_cfg = model_cfg.get("oscillator", {})

        if stage in ("mlp", "all"):
            ecg_to_osc_mlp, mlp_losses = train_mlp_on_frozen_brain(
                heart_model,
                brain_params,
                Sc_reduced,
                N,
                target_signal,
                t,
                device,
                ecg_dim=ecg_cfg.get("ecg_dim", 50),
                n_vns=ecg_cfg.get("n_vns", 64),
                hidden_dim=ecg_cfg.get("hidden_dim", 64),
                T=osc_cfg.get("T", 2.0),
                dt=0.01,
                omega1_hz=dyn_heart.get("omega1_hz", 1.0),
                omega2_hz=dyn_heart.get("omega2_hz", 1.2),
                num_epochs=train_cfg.get("mlp_epochs", 100),
                lr=train_cfg.get("mlp_lr", 1e-2),
                log_interval=train_cfg.get("log_interval", 20),
            )
        else:
            from models.ecg_to_brain import ECGToOscillatorMLP
            ecg_to_osc_mlp = ECGToOscillatorMLP(
                ecg_dim=ecg_cfg.get("ecg_dim", 50),
                N_VNS=ecg_cfg.get("n_vns", 64),
                hidden_dim=ecg_cfg.get("hidden_dim", 64),
                output_dim=N,
                T=osc_cfg.get("T", 2.0),
                device=device,
            ).to(device)
            mlp_losses = []

        model_final = TorchRevHopfNetwork(
            mu=1.0,
            eta_omega=0.0,
            eta_alpha=0.0,
            eta_theta=0.0,
            D_tensor=D_tensor,
            t_eval=t_eval,
            N=N,
            Sc=Sc_reduced,
            mlp_model=ecg_to_osc_mlp,
            hidden_repr=hidden_repr,
            device=device,
        )
        r_final, phi_final, theta_final, omega_final, alpha_final, rcos_phi_final = model_final.solve(
            brain_params["r"], brain_params["phi"], brain_params["theta"],
            brain_params["omega"], brain_params["alpha"],
        )

        if stage in ("feedback", "all"):
            heart_model, feedback_mlp, feedback_losses = train_feedback_loop(
                heart_model,
                rcos_phi_final,
                ecg_processed,
                device,
                T=t_duration,
                dt=1 / fs,
                omega1_hz=dyn_heart.get("omega1_hz", 1.0),
                omega2_hz=dyn_heart.get("omega2_hz", 1.2),
                num_epochs=train_cfg.get("feedback_epochs", 5000),
                lr=train_cfg.get("feedback_lr", 1e-3),
                log_interval=train_cfg.get("log_interval", 500),
            )
        else:
            feedback_mlp = FeedbackMLP().to(device)
            feedback_losses = []

        heart_model.eval()
        feedback_mlp.eval()

        with torch.no_grad():
            sim_baseline = simulate_coupled_oscillators_numpy(
                T=t_duration, dt=1 / fs,
                omega1_hz=dyn_heart.get("omega1_hz", 1.0),
                omega2_hz=dyn_heart.get("omega2_hz", 1.2),
            )
            pred_ecg_baseline = heart_model(
                torch.tensor(sim_baseline, dtype=torch.float32, device=device)
            ).cpu().numpy().flatten()

            rcos_phi_detach = rcos_phi_final.detach()
            modulation = feedback_mlp(rcos_phi_detach.unsqueeze(-1))
            heart_osc_torch = HeartOscillatorTorch(
                omega1_hz=dyn_heart.get("omega1_hz", 1.0),
                omega2_hz=dyn_heart.get("omega2_hz", 1.2),
            ).to(device)
            heart_traj_feedback = heart_osc_torch(T=t_duration, dt=1 / fs, modulation=modulation)
            pred_ecg_feedback = heart_model(heart_traj_feedback).cpu().numpy().flatten()

            P_out_baseline = torch.sum(alpha_final * r_final * torch.cos(phi_final), dim=1).cpu().numpy()

        target_ecg = ecg_processed[::10]

        plot_full_results(
            brain_losses=brain_losses,
            mlp_losses=mlp_losses,
            feedback_losses=feedback_losses,
            target_ecg=target_ecg,
            predicted_ecg_baseline=pred_ecg_baseline,
            predicted_ecg_feedback=pred_ecg_feedback,
            target_eeg=D_function(t),
            P_out_baseline=P_out_baseline,
            t=t,
            t_duration=t_duration,
            target_idx=target_idx,
            output_dir=figures_dir,
        )

        np.savez(
            output_dir / f"results_idx{target_idx}.npz",
            brain_losses=brain_losses,
            mlp_losses=mlp_losses,
            feedback_losses=feedback_losses,
            rcos_phi_final=rcos_phi_final.detach().cpu().numpy(),
            P_out_baseline=P_out_baseline,
            predicted_ecg_baseline=pred_ecg_baseline,
            predicted_ecg_feedback=pred_ecg_feedback,
            target_ecg=target_ecg,
            target_eeg=D_function(t),
        )

    logger.info("Pipeline complete. Check %s and %s", figures_dir, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument(
        "--stage",
        choices=["heart", "brain", "mlp", "feedback", "all"],
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume")
    parser.add_argument("--ckpt_interval", action="store_true", help="Save checkpoints")
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parent
    cfg_path = proj_root / args.config
    cfg = load_config(cfg_path)
    run_pipeline(cfg, args)


if __name__ == "__main__":
    main()
