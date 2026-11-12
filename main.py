"""
ShieldQKD command-line entry point.

    python main.py generate-dataset [--test]        # 24 000 BBM92 sessions -> data/
    python main.py train [--search grid|random]     # XGBoost / RF / SVM -> models/
    python main.py evaluate                         # metrics + figures -> results/
    python main.py ablation                         # feature-group ablation
    python main.py session --attack intercept_resend --rate 0.15   # one secured session
    python main.py demo                             # the 3-step evaluator demo (Section 13)
    python main.py dashboard                        # Streamlit UI
    python main.py test                             # pytest
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import settings as S  # noqa: E402

ATTACK_PARAM_FLAG = {
    "intercept_resend": "interception_rate", "pns": "multi_photon_prob",
    "beam_splitting": "splitting_ratio", "time_shift": "shift_prob",
    "trojan_horse": "probe_rate", "quantum_cloning": "clone_fidelity", "fake_state": "fake_rate",
}
COLORS = {"PROCEED": "\033[92m", "CAUTION": "\033[93m", "WARN": "\033[93m",
          "ABORT_MITM": "\033[91m", "ABORT_ATTACK": "\033[91m"}
RESET = "\033[0m"


def _print_result(out: dict, verbose: bool = True):
    ev = out["evaluation"]
    s = out["session"]["metadata"]
    c = COLORS.get(ev["decision"], "")
    print(f"  Protocol/backend : {out['protocol']} / {s['backend']}")
    print(f"  PQC signature    : {'VERIFIED' if out['pqc']['pqc_verified'] else 'FAILED - tampering detected'}")
    if ev.get("qber_stats"):
        q = ev["qber_stats"]
        print(f"  Average QBER     : {q['mean_qber'] * 100:.2f}%  (windows: {q['n_windows']}, "
              f"loss {q['photon_loss_rate'] * 100:.1f}%)")
        print(f"  Fixed 11% rule   : {ev['threshold_verdict']['verdict']}")
        top = sorted(ev["all_probabilities"].items(), key=lambda kv: -kv[1])[:3]
        print("  ML classifier    : " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in top))
    print(f"  Circuit breaker  : {c}{ev['decision']}{RESET}"
          + (f"  [{ev['attack_class']}, {ev['confidence'] * 100:.1f}%]" if ev.get("attack_class") else ""))
    if ev.get("shap_values"):
        print("  SHAP top features: " + ", ".join(f"{n} ({v:+.2f})" for n, v, _ in ev["shap_values"]["top_features"]))
    print(f"  Response         : {ev['recommended_action']}")
    print(f"  Key material     : {out['key_status']}")
    if ev.get("report_paths", {}).get("path_txt"):
        print(f"  Forensics report : {ev['report_paths']['path_txt']}")
    if verbose and ev["decision"] not in ("PROCEED",):
        print()
        print(ev["forensics_report"])


def cmd_session(args):
    from security.pipeline import run_shieldqkd
    params = {}
    if args.attack and args.rate is not None:
        params[ATTACK_PARAM_FLAG[args.attack]] = args.rate
    if args.attack == "time_shift" and args.burst_len:
        params["burst_len"] = args.burst_len
    if args.attack == "fake_state" and args.basis_bias is not None:
        params["basis_bias"] = args.basis_bias
    out = run_shieldqkd(n_qubits=args.qubits, noise_model=args.noise,
                        noise_prob=args.noise_prob, attack_type=args.attack, attack_params=params,
                        mitm=args.mitm, backend=args.backend, seed=args.seed,
                        pdf_report=args.pdf)
    print("\n=== ShieldQKD session ===")
    _print_result(out, verbose=args.verbose)
    if (args.message or args.encrypt_file) and not out["key_id"]:
        print("\n  Nothing encrypted: no key was released for this session.")
        return
    if out["key_id"] and args.message:
        from crypto import FileVault
        vault = FileVault(out["key_manager"])
        blob = vault.encrypt_bytes(args.message.encode(), out["key_id"])
        print(f"\n  AES-256-GCM ciphertext ({len(blob)} B): {blob.hex()[:96]}...")
        print(f"  Bob decrypts      : {vault.decrypt_bytes(blob).decode()!r}")
        print(f"  Key destroyed     : {out['key_manager'].is_destroyed(out['key_id'])}")
    elif out["key_id"] and args.encrypt_file:
        import hashlib
        from pathlib import Path
        from crypto import FileVault
        src = Path(args.encrypt_file)
        vault = FileVault(out["key_manager"])
        enc = vault.encrypt_file(str(src), out["key_id"])                    # Alice
        dec = src.with_name(f"{src.stem}.decrypted{src.suffix}")
        vault.decrypt_file(enc, src.name, str(dec))                          # Bob
        same = hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(dec.read_bytes()).digest()
        print(f"\n  Encrypted (Alice) : {enc}  ({os.path.getsize(enc)} B, AES-256-GCM)")
        print(f"  Decrypted (Bob)   : {dec}")
        print(f"  Byte-identical    : {same}")
        print(f"  Key destroyed     : {out['key_manager'].is_destroyed(out['key_id'])}")


def cmd_demo(args):
    """Section 13 - the 3-minute live demonstration."""
    from crypto import FileVault
    from security.pipeline import run_shieldqkd

    banner = lambda t: print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)  # noqa: E731
    banner("DEMO STEP 1: CLEAN BASELINE - no attack, normal channel noise")
    out = run_shieldqkd(noise_model="depolarizing", noise_prob=0.02, backend=args.backend, seed=101)
    _print_result(out, verbose=False)
    if out["key_id"]:
        vault = FileVault(out["key_manager"])
        blob = vault.encrypt_bytes(b"Launch codes: 0000", out["key_id"])
        print(f"  Encrypted message: {blob.hex()[:64]}...  -> decrypted: {vault.decrypt_bytes(blob)!r}")

    banner("DEMO STEP 2: CLASSICAL MITM - attacker tampers with the basis announcement")
    out = run_shieldqkd(noise_model="depolarizing", noise_prob=0.02, mitm="tamper",
                        backend=args.backend, seed=102)
    _print_result(out, verbose=False)
    print("  PQC log: " + " | ".join(out["pqc"]["log"][-2:]))

    banner("DEMO STEP 3: STEALTHY QUANTUM ATTACK - intercept-resend on 15% of qubits")
    out = run_shieldqkd(noise_model="depolarizing", noise_prob=0.01, attack_type="intercept_resend",
                        attack_params={"interception_rate": 0.15}, backend=args.backend, seed=103,
                        pdf_report=True)
    _print_result(out, verbose=False)
    ev = out["evaluation"]
    if ev.get("report_paths", {}).get("path_pdf"):
        print(f"  Forensics PDF    : {ev['report_paths']['path_pdf']}")
    tv = ev["threshold_verdict"]
    if tv and not tv["flags_attack"] and ev["decision"] in ("ABORT_ATTACK", "WARN"):
        print(f"\nConventional 11% threshold said '{tv['verdict']}' -> a FALSE NEGATIVE; "
              f"ShieldQKD caught {ev['attack_class']} at {ev['confidence'] * 100:.1f}% confidence.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="ShieldQKD - PQC-authenticated QKD with ML threat intelligence")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-dataset", help="generate the 24k-session dataset")
    g.add_argument("--sessions-per-class", type=int, default=S.SESSIONS_PER_CLASS)
    g.add_argument("--test", action="store_true")
    g.add_argument("--n-jobs", type=int, default=-1)

    t = sub.add_parser("train", help="train XGBoost / RF / SVM")
    t.add_argument("--search", choices=["grid", "random"], default="grid")
    t.add_argument("--n-iter", type=int, default=20)
    t.add_argument("--runs", type=int, default=S.N_EVAL_RUNS)
    t.add_argument("--models", nargs="+", default=["xgboost", "random_forest", "svm_rbf"])

    sub.add_parser("evaluate", help="re-evaluate saved models and regenerate figures")
    sub.add_parser("ablation", help="feature-group ablation study")

    s = sub.add_parser("session", help="run one fully secured BBM92 session")
    s.add_argument("--backend", choices=S.BACKENDS, default="numpy")
    s.add_argument("--qubits", type=int, default=S.DEFAULT_N_QUBITS)
    s.add_argument("--noise", choices=S.NOISE_MODELS, default=None)
    s.add_argument("--noise-prob", type=float, default=0.0)
    s.add_argument("--attack", choices=S.ATTACK_TYPES, default=None)
    s.add_argument("--rate", type=float, default=None, help="main attack parameter")
    s.add_argument("--burst-len", type=int, default=None)
    s.add_argument("--basis-bias", type=float, default=None)
    s.add_argument("--mitm", choices=["tamper", "forge", "replay"], default=None)
    s.add_argument("--seed", type=int, default=None)
    enc_grp = s.add_mutually_exclusive_group()          # one one-time key -> one payload
    enc_grp.add_argument("--message", default=None, help="encrypt this message if the key is accepted")
    enc_grp.add_argument("--encrypt-file", default=None, metavar="PATH",
                         help="encrypt this file (-> PATH.sqkd) if the key is accepted; Bob's "
                              "decrypted copy is written next to it")
    s.add_argument("--pdf", action="store_true", help="also write a PDF forensics report")
    s.add_argument("-v", "--verbose", action="store_true")

    d = sub.add_parser("demo", help="Section 13 live demonstration")
    d.add_argument("--backend", choices=S.BACKENDS, default="qiskit")

    sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    sub.add_parser("test", help="run the test-suite")

    args = ap.parse_args(argv)
    if args.cmd == "generate-dataset":
        from training.generate_dataset import main as gen
        gen(["--sessions-per-class", str(args.sessions_per_class), "--n-jobs", str(args.n_jobs)]
            + (["--test"] if args.test else []))
    elif args.cmd == "train":
        from training.train_models import main as tr
        tr(["--search", args.search, "--n-iter", str(args.n_iter), "--runs", str(args.runs),
            "--models", *args.models])
    elif args.cmd == "evaluate":
        from training.evaluate import main as ev
        ev()
    elif args.cmd == "ablation":
        from training.ablation_study import main as ab
        ab([])
    elif args.cmd == "session":
        cmd_session(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    elif args.cmd == "dashboard":
        subprocess.call([sys.executable, "-m", "streamlit", "run",
                         os.path.join(ROOT, "dashboard", "app.py")], cwd=ROOT)
    elif args.cmd == "test":
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT))


if __name__ == "__main__":
    main()
