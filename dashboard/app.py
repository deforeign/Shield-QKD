"""
ShieldQKD Streamlit dashboard.

    streamlit run dashboard/app.py      (or: python main.py dashboard)

Tabs: (1) Live Demo  (2) Training Results  (3) Forensics Reports
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings as S
from crypto import FileVault, KeyManager
from dashboard.components.attack_alert import render_alert
from dashboard.components.forensics_viewer import render_forensics
from dashboard.components.qber_live_chart import render_live
from dashboard.components.shap_waterfall import probability_figure, shap_waterfall_figure
from security.response_router import ResponseRouter

st.set_page_config(page_title="ShieldQKD", page_icon="🛡️", layout="wide")

ATTACK_LABELS = {
    None: "None (clean channel)", "intercept_resend": "Intercept-Resend", "pns": "Photon Number Splitting",
    "beam_splitting": "Beam Splitting", "time_shift": "Time-Shift", "trojan_horse": "Trojan Horse",
    "quantum_cloning": "Quantum Cloning", "fake_state": "Fake State",
}
PARAM_SPECS = {   # attack -> list of (param, label, min, max, default, step)
    "intercept_resend": [("interception_rate", "Interception rate", 0.0, 1.0, 0.15, 0.01)],
    "pns": [("multi_photon_prob", "Multi-photon probability", 0.0, 0.3, 0.15, 0.01)],
    "beam_splitting": [("splitting_ratio", "Splitting ratio", 0.0, 0.5, 0.25, 0.01)],
    "time_shift": [("shift_prob", "Shift probability (per burst)", 0.0, 0.3, 0.15, 0.01),
                   ("burst_len", "Burst length (qubits)", 5, 20, 10, 1)],
    "trojan_horse": [("probe_rate", "Probe rate", 0.0, 0.3, 0.15, 0.01)],
    "quantum_cloning": [("clone_fidelity", "Clone fidelity", 0.5, 0.85, 0.70, 0.01)],
    "fake_state": [("fake_rate", "Fake rate", 0.0, 0.5, 0.25, 0.01),
                   ("basis_bias", "Basis bias (P rectilinear)", 0.0, 1.0, 0.9, 0.05)],
}
SCENARIOS = {
    "Custom": None,
    "Demo 1 - clean baseline": dict(attack=None, noise="depolarizing", noise_prob=0.02, mitm=False),
    "Demo 2 - classical MITM": dict(attack=None, noise="depolarizing", noise_prob=0.02, mitm=True),
    "Demo 3 - stealthy intercept-resend (15%)": dict(attack="intercept_resend", noise="depolarizing",
                                                     noise_prob=0.01, mitm=False,
                                                     params={"interception_rate": 0.15}),
}


@st.cache_resource(show_spinner="Loading XGBoost model + SHAP explainer ...")
def load_breaker():
    from security.circuit_breaker import CircuitBreaker
    if not os.path.exists(S.XGBOOST_MODEL_PATH):
        return None
    return CircuitBreaker(str(S.XGBOOST_MODEL_PATH))


@st.cache_data
def load_metrics():
    if not os.path.exists(S.METRICS_PATH):
        return None
    with open(S.METRICS_PATH) as fh:
        return json.load(fh)


def key_manager() -> KeyManager:
    if "km" not in st.session_state:
        st.session_state.km = KeyManager()
    return st.session_state.km


# ==========================================================================
st.title("🛡️ ShieldQKD")
st.caption("Hybrid PQC-authenticated QKD with ML-powered temporal threat intelligence - "
           "Lamport OTS signatures · BBM92 entanglement-based QKD · 63-feature temporal QBER · XGBoost · SHAP")

breaker = load_breaker()
if breaker is None:
    st.error("No trained model found. Run `python main.py generate-dataset --test` and "
             "`python main.py train` first.")

tab_live, tab_train, tab_forensics = st.tabs(
    ["⚡ Live Demo", "📊 Training Results", "🔎 Forensics Reports"])

# --------------------------------------------------------------------------
# Sidebar - session configuration
# --------------------------------------------------------------------------
def _param_key(attack_name: str, param: str) -> str:
    return f"w_param_{attack_name}_{param}"


def _init_widget_state():
    """Defaults for every scenario-controlled widget (set once per browser session)."""
    ss = st.session_state
    ss.setdefault("w_noise", "depolarizing")
    ss.setdefault("w_noise_prob", 0.02)
    ss.setdefault("w_attack", None)
    ss.setdefault("w_mitm", False)
    for atk, specs in PARAM_SPECS.items():
        for name, _label, _lo, _hi, default, _step in specs:
            ss.setdefault(_param_key(atk, name), default)


def apply_scenario():
    """Write the chosen scenario into the widgets' state. Without this, Streamlit
    keeps a widget's previous value whenever its default happens to be unchanged
    (e.g. a MITM switch left ON from Demo 2 would leak into Demo 1)."""
    sc = SCENARIOS[st.session_state.w_scenario]
    if not sc:
        return                                            # "Custom": keep current settings
    ss = st.session_state
    ss.w_noise = sc["noise"]
    ss.w_noise_prob = float(sc["noise_prob"])
    ss.w_attack = sc["attack"]
    ss.w_mitm = bool(sc["mitm"])
    for atk, specs in PARAM_SPECS.items():                # reset every attack parameter
        for name, _label, _lo, _hi, default, _step in specs:
            ss[_param_key(atk, name)] = default
    for name, val in sc.get("params", {}).items():
        ss[_param_key(sc["attack"], name)] = val


_init_widget_state()

with st.sidebar:
    st.header("Session setup")
    scenario = st.selectbox("Scenario", list(SCENARIOS), key="w_scenario", on_change=apply_scenario)
    st.caption("Protocol: **BBM92** (entanglement-based)")
    backend = "qiskit"
    st.caption("Quantum backend: **Qiskit-Aer** (real quantum circuits)")
    n_qubits = st.slider("Qubits sent", 2000, 20000, S.DEFAULT_N_QUBITS, 500)
    noise_opts = [None] + S.NOISE_MODELS
    noise_model = st.selectbox("Noise channel", noise_opts, key="w_noise",
                               format_func=lambda v: "none" if v is None else v)
    noise_prob = st.slider("Noise probability", 0.0, 0.10, step=0.005, key="w_noise_prob")
    channel_loss = st.slider("Baseline channel loss", 0.0, 0.05, S.DEFAULT_CHANNEL_LOSS, 0.005)
    st.subheader("Eve")
    attack_keys = list(ATTACK_LABELS)
    attack = st.selectbox("Quantum attack", attack_keys, key="w_attack",
                          format_func=lambda k: ATTACK_LABELS[k])
    attack_params = {}
    for name, label, lo, hi, default, step in PARAM_SPECS.get(attack, []):
        k = _param_key(attack, name)
        if isinstance(step, int):
            attack_params[name] = st.slider(label, lo, hi, step=step, key=k)
        else:
            attack_params[name] = st.slider(label, float(lo), float(hi), step=float(step), key=k)
    mitm = st.toggle("Classical MITM on basis announcement", key="w_mitm")
    sc_active = SCENARIOS[scenario]
    if sc_active and (mitm != sc_active["mitm"] or attack != sc_active["attack"]):
        st.caption(f"⚠️ Settings differ from the '{scenario}' preset - re-select it to reset.")
    mitm_mode = st.selectbox("MITM technique", ["tamper", "forge", "replay"], disabled=not mitm)
    seed_txt = st.text_input("Seed (blank = random)", "")
    seed = int(seed_txt) if seed_txt.strip().isdigit() else None
    animate = st.checkbox("Animate QBER windows", value=True)
    resample = st.checkbox("Re-classify on uncertain verdicts", value=True)
    run = st.button("▶ Run secured QKD session", type="primary", use_container_width=True,
                    disabled=breaker is None)

# --------------------------------------------------------------------------
# Tab 1 - Live demo
# --------------------------------------------------------------------------
with tab_live:
    if run and breaker is not None:
        from security.pipeline import run_shieldqkd
        with st.spinner("Running BBM92 on Qiskit-Aer ..."):
            out = run_shieldqkd(n_qubits=n_qubits, noise_model=noise_model,
                                noise_prob=noise_prob, attack_type=attack, attack_params=attack_params,
                                channel_loss=channel_loss, mitm=mitm_mode if mitm else None,
                                backend=backend, seed=seed, breaker=breaker,
                                key_manager=key_manager(), resample=resample, pdf_report=True)
        st.session_state.last = out
        st.session_state.animate_pending = animate

    out = st.session_state.get("last")
    if out is None:
        st.info("Configure a session in the sidebar (or pick a demo scenario) and press **Run**.")
        st.markdown("""
**What happens on Run**
1. Alice and Bob exchange Lamport one-time public keys (PQC handshake)
2. A source emits entangled pairs (BBM92); Alice measures her photon, Bob's photon crosses the
   noisy, possibly attacked channel (simulated with Qiskit-Aer circuits)
3. Basis announcements are signed + verified (tampering -> **ABORT_MITM**)
4. The sifted key is cut into W = 50 windows -> temporal QBER -> 63 features -> XGBoost
5. Circuit breaker: PROCEED / CAUTION / WARN / ABORT_ATTACK, SHAP explanation, forensics report
6. On PROCEED an AES-256 key is derived and can encrypt a message (one-time use)
""")
    else:
        ev, md = out["evaluation"], out["session"]["metadata"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PQC signature", "VERIFIED" if out["pqc"]["pqc_verified"] else "FAILED")
        if ev.get("qber_stats"):
            q = ev["qber_stats"]
            c2.metric("Average QBER", f"{q['mean_qber'] * 100:.2f}%")
            c3.metric("Fixed 11% rule", "ABORT" if ev["threshold_verdict"]["flags_attack"] else "All clear")
            c4.metric("ML verdict", ev.get("predicted_label", "-"), f"{ev['confidence'] * 100:.1f}% conf.",
                      delta_color="off")
        else:
            c2.metric("Average QBER", "n/a")
            c3.metric("Fixed 11% rule", "n/a")
            c4.metric("ML verdict", "skipped")
        c5.metric("Sifted bits", f"{md['n_sifted']}", f"loss {md['photon_loss_rate'] * 100:.1f}%",
                  delta_color="off")
        render_alert(st, ev, out["key_status"])
        if ev.get("qber_stats") and not ev["threshold_verdict"]["flags_attack"] \
                and ev["decision"] in ("ABORT_ATTACK", "WARN"):
            st.warning(f"The conventional 11% threshold would have said **all clear** "
                       f"(QBER {ev['qber_stats']['mean_qber'] * 100:.2f}%) - a false negative. "
                       f"Temporal analysis caught **{ev['attack_class']}**.")

        if ev.get("qber_sequence") is not None:
            left, right = st.columns([1.5, 1])
            with left:
                ph = st.empty()
                render_live(st, ph, ev["qber_sequence"], ev.get("qber_rect_sequence"),
                            ev.get("qber_diag_sequence"),
                            animate=st.session_state.pop("animate_pending", False))
            with right:
                st.plotly_chart(probability_figure(ev["all_probabilities"]), use_container_width=True)
            if ev.get("shap_values"):
                l2, r2 = st.columns([1.5, 1])
                with l2:
                    st.plotly_chart(shap_waterfall_figure(ev["shap_values"]), use_container_width=True)
                with r2:
                    st.markdown("**What the top features mean**")
                    for it in ev["shap_values"]["interpretations"]:
                        st.markdown(f"- `{it['feature']}` = {it['actual_value']:.4g} "
                                    f"{it['direction']} *{ev['shap_values']['explained_class']}* - "
                                    f"{it['meaning']}")
        else:
            st.error("Classical channel tampering detected before any key analysis. "
                     "All key material was destroyed.")
            st.code("\n".join(out["pqc"]["log"]), language="text")

        with st.expander("Forensics report", expanded=ev["decision"] not in ("PROCEED",)):
            st.code(ev["forensics_report"], language="text")
            rp = ev.get("report_paths") or {}
            if rp.get("path_pdf") and os.path.exists(rp["path_pdf"]):
                with open(rp["path_pdf"], "rb") as fh:
                    st.download_button("Download forensics PDF", fh.read(),
                                       file_name=os.path.basename(rp["path_pdf"]))
        with st.expander("Session timings & PQC audit"):
            st.json({k: round(v, 3) for k, v in out["timings"].items()})
            st.code("\n".join(out["pqc"]["log"]), language="text")
            st.caption(f"Lamport signature size: 8 KiB · public key 16 KiB · "
                       f"Alice key fingerprint {out['handshake']['alice_fingerprint']}")

        st.subheader("Application layer - AES-256-GCM")
        km = key_manager()
        vr = st.session_state.get("vault_result")
        if vr and vr["key_id"] == out["key_id"]:            # result of this session's one-time key
            if vr["kind"] == "message":
                st.code(f"ciphertext ({len(vr['blob'])} B): {vr['blob'].hex()}", language="text")
                st.success(f"Bob decrypted: {vr['plain'].decode()!r}")
            else:
                st.success(f"'{vr['name']}' ({len(vr['plain']):,} B) encrypted by Alice and decrypted by Bob "
                           f"- byte-identical: {vr['identical']}")
                d1, d2 = st.columns(2)
                d1.download_button("Download encrypted file (.sqkd)", vr["blob"],
                                   file_name=vr["name"] + ".sqkd", mime="application/octet-stream")
                d2.download_button("Download Bob's decrypted copy", vr["plain"], file_name=vr["name"])
            st.caption(f"Key {out['key_id']} was used once and has been destroyed (forward secrecy). "
                       "Run a new session for a fresh key.")
        elif out["key_id"] and not km.is_destroyed(out["key_id"]):
            kind = st.radio("Encrypt", ["Message", "File"], horizontal=True,
                            help="Each QKD key is one-time: it encrypts exactly one payload.")
            if kind == "Message":
                msg = st.text_input("Message to encrypt with the QKD key", "Meet at the quantum lab at 09:00")
                if st.button("Encrypt (Alice) → decrypt (Bob)"):
                    vault = FileVault(km)
                    blob = vault.encrypt_bytes(msg.encode(), out["key_id"])
                    st.session_state.vault_result = {"key_id": out["key_id"], "kind": "message",
                                                     "blob": blob, "plain": vault.decrypt_bytes(blob)}
                    st.rerun()
            else:
                up = st.file_uploader("File to encrypt with the QKD key")
                if up is not None and st.button("Encrypt file (Alice) → decrypt (Bob)"):
                    data = up.getvalue()
                    vault = FileVault(km)
                    aad = up.name.encode()                  # file name is authenticated too
                    blob = vault.encrypt_bytes(data, out["key_id"], "alice", aad=aad)
                    plain = vault.decrypt_bytes(blob, "bob", aad=aad)
                    st.session_state.vault_result = {"key_id": out["key_id"], "kind": "file",
                                                     "name": up.name, "blob": blob, "plain": plain,
                                                     "identical": plain == data}
                    st.rerun()
        elif out["key_id"]:
            st.caption(f"Key {out['key_id']} has been used once and destroyed (forward secrecy). "
                       "Run a new session for a fresh key.")
        else:
            st.caption(f"No key: {out['key_status']}")

# --------------------------------------------------------------------------
# Tab 2 - Training results
# --------------------------------------------------------------------------
with tab_train:
    m = load_metrics()
    if m is None:
        st.info("No training metrics yet - run `python main.py train`.")
    else:
        rows = []
        for name, e in m["models"].items():
            r = e.get("runs", {})
            rows.append({
                "model": name,
                "CV accuracy": e["cv"].get("cv_accuracy"),
                "test accuracy": e["test"]["accuracy"],
                f"accuracy ({r.get('n_runs', 0)} runs)": (f"{r['accuracy_mean'] * 100:.2f} ± {r['accuracy_std'] * 100:.2f}%"
                                                         if r else "-"),
                f"macro-F1 ({r.get('n_runs', 0)} runs)": (f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}"
                                                         if r else "-"),
                "macro ROC-AUC": e["test"]["macro_roc_auc"],
                "FNR (attack vs normal)": e["test"]["binary"]["fnr"],
                "independent test acc": e.get("independent_test", {}).get("accuracy"),
                "best params": json.dumps(e["cv"].get("best_params")),
            })
        st.subheader(f"Classifier performance - {m['dataset']['n_sessions']} sessions, "
                     f"{m['search']} search")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        comp = m.get("comparison", {})
        if comp:
            names = list(comp)
            fig = go.Figure()
            fig.add_bar(x=names, y=[comp[n]["accuracy"] * 100 for n in names], name="Detection accuracy %",
                        marker_color="#2b7bba", text=[f"{comp[n]['accuracy'] * 100:.1f}" for n in names])
            fig.add_bar(x=names, y=[comp[n]["fnr"] * 100 for n in names], name="False negative rate %",
                        marker_color="#d1495b", text=[f"{comp[n]['fnr'] * 100:.1f}" for n in names])
            fig.update_layout(barmode="group", title="Fixed threshold vs CUSUM vs ML (attack-vs-normal)",
                              height=380, yaxis=dict(range=[0, 105]), margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        sel = st.selectbox("Model", list(m["models"]))
        test = m["models"][sel]["test"]
        labels = list(test["per_class_f1"])
        a, b = st.columns(2)
        with a:
            cm = np.array(test["confusion_matrix"])
            cmn = cm / cm.sum(axis=1, keepdims=True)
            fig = go.Figure(go.Heatmap(z=cmn, x=labels, y=labels, colorscale="Blues", text=cm,
                                       texttemplate="%{text}", zmin=0, zmax=1))
            fig.update_layout(title=f"Confusion matrix - {sel}", height=460,
                              yaxis=dict(autorange="reversed", title="true"), xaxis_title="predicted",
                              margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = go.Figure()
            for cls, curve in test["roc_curves"].items():
                fig.add_scatter(x=curve["fpr"], y=curve["tpr"], mode="lines",
                                name=f"{cls} ({test['roc_auc_per_class'][cls]:.3f})")
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                            showlegend=False)
            fig.update_layout(title=f"One-vs-rest ROC - macro AUC {test['macro_roc_auc']:.4f}",
                              height=460, xaxis_title="FPR", yaxis_title="TPR", margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Per-class F1**")
        st.dataframe(pd.DataFrame([test["per_class_f1"]]), use_container_width=True, hide_index=True)

        if m.get("global_shap_importance"):
            imp = pd.DataFrame(m["global_shap_importance"]).head(20)
            fig = go.Figure(go.Bar(x=imp["mean_abs_shap"][::-1], y=imp["feature"][::-1], orientation="h",
                                   marker_color="#2b7bba"))
            fig.update_layout(title="Global feature importance (mean |SHAP|, XGBoost, top 20)",
                              height=520, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        abl = os.path.join(S.FIGURES_DIR, "ablation.png")
        if os.path.exists(abl):
            st.image(abl, caption="Feature-group ablation study")
        base = m.get("threshold_baseline", {})
        if base.get("detection_rate_per_class"):
            st.markdown("**Fixed 11% threshold - detection rate per attack type**")
            st.dataframe(pd.DataFrame([{k: f"{v * 100:.1f}%" for k, v in base["detection_rate_per_class"].items()}]),
                         use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Tab 3 - Forensics
# --------------------------------------------------------------------------
with tab_forensics:
    render_forensics(st)
    st.markdown("**Response routing table**")
    st.dataframe(pd.DataFrame(ResponseRouter.table(), columns=["Attack class", "Recommended response"]),
                 use_container_width=True, hide_index=True)
