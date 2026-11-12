"""
ShieldQKD Streamlit dashboard.

    streamlit run dashboard/app.py      (or: python main.py dashboard)

Tabs: (1) Live Demo  (2) Training Results  (3) Forensics Reports
"""
from __future__ import annotations

import json
import os
import sys
from html import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings as S
from crypto import FileVault, KeyManager
from dashboard import theme as T
from dashboard.components.attack_alert import (render_instruments, render_key_status, render_line,
                                               render_note, render_stages, render_steps, render_verdict,
                                               session_stages)
from dashboard.components.forensics_viewer import render_forensics
from dashboard.components.qber_live_chart import render_live
from dashboard.components.shap_waterfall import probability_figure, shap_waterfall_figure
from security.response_router import ResponseRouter

st.set_page_config(page_title="ShieldQKD", page_icon="🛡️", layout="wide")
T.apply(st)

CLEAN = "clean"           # widget value for "no attack" (None would read as "nothing selected")
ATTACK_LABELS = {
    CLEAN: "None (clean channel)", "intercept_resend": "Intercept-Resend", "pns": "Photon Number Splitting",
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


@st.cache_resource(show_spinner="Loading the attack detector ...")
def load_breaker():
    from security.circuit_breaker import CircuitBreaker
    if not os.path.exists(S.XGBOOST_MODEL_PATH):
        return None
    return CircuitBreaker(str(S.XGBOOST_MODEL_PATH))


@st.cache_data
def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def key_manager() -> KeyManager:
    if "km" not in st.session_state:
        st.session_state.km = KeyManager()
    return st.session_state.km


def section(title: str, sub: str | None = None) -> None:
    st.markdown(f'<div class="sq-section">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sq-sub">{sub}</div>', unsafe_allow_html=True)


# ==========================================================================
st.markdown(
    '<div class="sq-head"><span class="mark">ShieldQKD</span>'
    '<span class="what">Quantum key distribution guarded by post-quantum signatures and a model '
    'that reads the error pattern for eavesdroppers.</span></div>',
    unsafe_allow_html=True)

breaker = load_breaker()
if breaker is None:
    st.error("No trained model found. Run `python main.py generate-dataset --test`, then "
             "`python main.py train`, and reload this page.")

tab_live, tab_train, tab_forensics = st.tabs(["Live Demo", "Training Results", "Forensics Reports"])

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
    ss.setdefault("w_attack", CLEAN)
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
    ss.w_attack = sc["attack"] or CLEAN
    ss.w_mitm = bool(sc["mitm"])
    for atk, specs in PARAM_SPECS.items():                # reset every attack parameter
        for name, _label, _lo, _hi, default, _step in specs:
            ss[_param_key(atk, name)] = default
    for name, val in sc.get("params", {}).items():
        ss[_param_key(sc["attack"], name)] = val


def group(title: str) -> None:
    st.markdown(f'<div class="sq-group">{title}</div>', unsafe_allow_html=True)


_init_widget_state()

with st.sidebar:
    st.header("Session setup")
    scenario = st.selectbox("Scenario", list(SCENARIOS), key="w_scenario", on_change=apply_scenario)
    backend = "qiskit"

    group("Channel")
    st.caption("BBM92 entangled pairs, simulated as Qiskit-Aer circuits")
    n_qubits = st.slider("Pairs sent", 2000, 20000, S.DEFAULT_N_QUBITS, 500)
    noise_opts = [None] + S.NOISE_MODELS
    noise_model = st.selectbox("Noise channel", noise_opts, key="w_noise",
                               format_func=lambda v: "none" if v is None else v.replace("_", " "))
    noise_prob = st.slider("Noise probability", 0.0, 0.10, step=0.005, key="w_noise_prob")
    channel_loss = st.slider("Baseline photon loss", 0.0, 0.05, S.DEFAULT_CHANNEL_LOSS, 0.005)

    group("Eavesdropper")
    attack_keys = list(ATTACK_LABELS)
    attack_choice = st.selectbox("Quantum attack", attack_keys, key="w_attack",
                                 format_func=lambda k: ATTACK_LABELS[k])
    attack = None if attack_choice == CLEAN else attack_choice
    attack_params = {}
    for name, label, lo, hi, default, step in PARAM_SPECS.get(attack, []):
        k = _param_key(attack, name)
        if isinstance(step, int):
            attack_params[name] = st.slider(label, lo, hi, step=step, key=k)
        else:
            attack_params[name] = st.slider(label, float(lo), float(hi), step=float(step), key=k)
    mitm = st.toggle("Tamper with the classical messages (MITM)", key="w_mitm")
    sc_active = SCENARIOS[scenario]
    if sc_active and (mitm != sc_active["mitm"] or attack != sc_active["attack"]):
        st.caption(f"Settings no longer match '{scenario}'. Choose it again to reset.")
    mitm_mode = st.selectbox("Tampering method", ["tamper", "forge", "replay"], disabled=not mitm)

    group("Options")
    seed_txt = st.text_input("Seed (leave blank for random)", "")
    seed = int(seed_txt) if seed_txt.strip().isdigit() else None
    animate = st.checkbox("Draw the error chart window by window", value=True)
    resample = st.checkbox("Re-check uncertain verdicts with extra sessions", value=True)
    run = st.button("Run secured session", type="primary", use_container_width=True,
                    disabled=breaker is None)

# --------------------------------------------------------------------------
# Tab 1 - Live demo
# --------------------------------------------------------------------------
with tab_live:
    if run and breaker is not None:
        from security.pipeline import run_shieldqkd
        with st.spinner("Sending entangled pairs through the channel ..."):
            out = run_shieldqkd(n_qubits=n_qubits, noise_model=noise_model,
                                noise_prob=noise_prob, attack_type=attack, attack_params=attack_params,
                                channel_loss=channel_loss, mitm=mitm_mode if mitm else None,
                                backend=backend, seed=seed, breaker=breaker,
                                key_manager=key_manager(), resample=resample, pdf_report=True)
        st.session_state.last = out
        st.session_state.animate_pending = animate

    out = st.session_state.get("last")
    if out is None:
        st.markdown("""<div class="sq-empty">
<h3>Run a session to see the verdict</h3>
<ol>
<li><b>Handshake.</b> Alice and Bob exchange one-time post-quantum public keys.</li>
<li><b>Transmission.</b> A source sends entangled pairs; Bob's photon crosses the noisy, possibly attacked channel.</li>
<li><b>Sifting.</b> They compare measurement bases over signed messages. Tampering stops the session here.</li>
<li><b>Analysis.</b> The kept bits are cut into 50-bit windows and the error pattern is measured 63 ways.</li>
<li><b>Verdict.</b> The detector accepts the key, withholds it, or destroys it, and explains why.</li>
<li><b>Use.</b> An accepted key encrypts one message or file with AES-256, then is destroyed.</li>
</ol>
<p style="margin:0.9rem 0 0;color:#9AA4BF">Pick a demo scenario in the sidebar, or set your own channel and attack, then press <b style="color:#E6EAF4;font-weight:500">Run secured session</b>.</p>
</div>""", unsafe_allow_html=True)
    else:
        ev, md = out["evaluation"], out["session"]["metadata"]
        km = key_manager()
        key_used = bool(out["key_id"]) and km.is_destroyed(out["key_id"])
        pqc_ok = out["pqc"]["pqc_verified"]
        analysed = bool(ev.get("qber_stats"))
        decision = ev["decision"]

        render_stages(st, session_stages(out, key_used))
        left, right = st.columns([0.38, 0.62], gap="large")

        # ---------------- left: the decision ----------------
        with left:
            render_verdict(st, ev)
            if ev.get("resampling_rounds"):
                n = ev["resampling_rounds"]
                render_line(st, f"The first verdict was uncertain, so {n} more session{'s' if n > 1 else ''} "
                                f"{'were' if n > 1 else 'was'} collected and combined.", T.MUTED)

            missed = False
            rows = [("Classical messages", "Verified" if pqc_ok else "Tampered", "Lamport signatures",
                     T.PROCEED if pqc_ok else T.ABORT)]
            if analysed:
                q = ev["qber_stats"]
                flags = ev["threshold_verdict"]["flags_attack"]
                missed = (not flags) and decision in ("ABORT_ATTACK", "WARN")
                rows += [
                    ("Average error", f"{q['mean_qber'] * 100:.2f}%", f"over {q['n_windows']} windows", None),
                    ("Old 11% rule", "Abort" if flags else "All clear",
                     "would have missed this" if missed else None, T.CAUTION if missed else None),
                    ("Model says", ev.get("predicted_label", "-"), f"{ev['confidence'] * 100:.1f}% confidence", None),
                ]
            else:
                rows += [("Average error", "Not measured", "stopped at sifting", T.MUTED),
                         ("Old 11% rule", "Not run", None, T.MUTED),
                         ("Model says", "Not run", None, T.MUTED)]
            rows.append(("Key bits kept", f"{md['n_sifted']:,}", f"{md['photon_loss_rate'] * 100:.1f}% photons lost",
                         None))
            render_instruments(st, rows)

            render_steps(st, "What happens next", ev.get("response", {}).get("details", []))

            if out["key_id"] and not key_used:
                render_key_status(st, "Ready", T.PROCEED, "AES-256 key derived after privacy amplification. "
                                  "It encrypts one message or file, then it is destroyed.")
            elif out["key_id"]:
                render_key_status(st, "Used and destroyed", T.MUTED,
                                  "This key encrypted one payload and was wiped. Run a new session for a fresh key.")
            elif decision in ("PROCEED", "CAUTION"):
                render_key_status(st, "Withheld", T.CAUTION,
                                  "The verdict was not certain enough to trust this key. Run another session."
                                  if decision == "CAUTION" else out["key_status"])
            else:
                render_key_status(st, "Destroyed", T.ABORT,
                                  "Every key bit was wiped, so an eavesdropper gains nothing usable.")

            if missed:
                render_line(st, f"The old 11% rule would have accepted this key (average error "
                                f"{ev['qber_stats']['mean_qber'] * 100:.2f}%). The way the errors were spread over "
                                f"time gave away {ev['attack_class']}.", T.CAUTION)

            # ---- use the key (only while this session's one-time key exists or was just used) ----
            vr = st.session_state.get("vault_result")
            if vr and vr["key_id"] == out["key_id"]:
                st.markdown('<div class="sq-block"><div class="ttl">Use the key</div></div>', unsafe_allow_html=True)
                if vr["kind"] == "message":
                    st.success(f"Bob decrypted: {vr['plain'].decode()!r}")
                    with st.expander("Ciphertext that crossed the network"):
                        st.code(vr["blob"].hex(), language="text", wrap_lines=True)
                else:
                    st.success(f"'{vr['name']}' ({len(vr['plain']):,} bytes) reached Bob intact: "
                               f"{'identical' if vr['identical'] else 'NOT identical'} to the original.")
                    st.download_button("Download encrypted file (.sqkd)", vr["blob"],
                                       file_name=vr["name"] + ".sqkd", mime="application/octet-stream",
                                       use_container_width=True)
                    st.download_button("Download Bob's decrypted copy", vr["plain"], file_name=vr["name"],
                                       use_container_width=True)
            elif out["key_id"] and not key_used:
                st.markdown('<div class="sq-block"><div class="ttl">Use the key</div>'
                            '<div class="expl">AES-256-GCM, one message or file per key.</div></div>',
                            unsafe_allow_html=True)
                kind = st.radio("Encrypt", ["Message", "File"], horizontal=True, label_visibility="collapsed")
                if kind == "Message":
                    msg = st.text_input("Message", "Meet at the quantum lab at 09:00")
                    if st.button("Encrypt and send to Bob", type="primary", use_container_width=True):
                        vault = FileVault(km)
                        blob = vault.encrypt_bytes(msg.encode(), out["key_id"])
                        st.session_state.vault_result = {"key_id": out["key_id"], "kind": "message",
                                                         "blob": blob, "plain": vault.decrypt_bytes(blob)}
                        st.rerun()
                else:
                    up = st.file_uploader("File")
                    if up is not None and st.button("Encrypt file and send to Bob", type="primary",
                                                    use_container_width=True):
                        data = up.getvalue()
                        vault = FileVault(km)
                        aad = up.name.encode()                  # file name is authenticated too
                        blob = vault.encrypt_bytes(data, out["key_id"], "alice", aad=aad)
                        plain = vault.decrypt_bytes(blob, "bob", aad=aad)
                        st.session_state.vault_result = {"key_id": out["key_id"], "kind": "file",
                                                         "name": up.name, "blob": blob, "plain": plain,
                                                         "identical": plain == data}
                        st.rerun()

        # ---------------- right: the evidence ----------------
        with right:
            if ev.get("qber_sequence") is not None:
                ph = st.empty()
                render_live(st, ph, ev["qber_sequence"], ev.get("qber_rect_sequence"),
                            ev.get("qber_diag_sequence"),
                            animate=st.session_state.pop("animate_pending", False))
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    st.plotly_chart(probability_figure(ev["all_probabilities"], height=330),
                                    use_container_width=True)
                with c2:
                    shap = ev.get("shap_values")
                    if shap:
                        items = "".join(
                            f'<div class="r"><b>{escape(it["meaning"][:1].upper() + it["meaning"][1:])}</b>'
                            f'<span>{escape(it["feature"])} = {it["actual_value"]:.4g}, '
                            f'{"supports" if it["shap_value"] > 0 else "argues against"} '
                            f'{escape(shap["explained_class"])}</span></div>'
                            for it in shap["interpretations"][:3])
                        st.markdown(f'<div class="sq-reasons"><div class="ttl">What gave it away</div>{items}</div>',
                                    unsafe_allow_html=True)
                    else:
                        q = ev["qber_stats"]
                        top = ev.get("predicted_label", "Normal")
                        items = (f'<div class="r"><b>No attack pattern matched</b><span>The model\'s top answer is '
                                 f'{escape(top)} at {ev["confidence"] * 100:.1f}%.</span></div>'
                                 f'<div class="r"><b>Errors stayed low</b><span>Average {q["mean_qber"] * 100:.2f}%, '
                                 f'well under the 11% rule.</span></div>'
                                 f'<div class="r"><b>Photons arrived as expected</b><span>'
                                 f'{q["photon_loss_rate"] * 100:.1f}% lost, in line with normal fibre loss.</span></div>')
                        st.markdown(f'<div class="sq-reasons"><div class="ttl">Why it looks clean</div>{items}</div>',
                                    unsafe_allow_html=True)
                if ev.get("shap_values"):
                    with st.expander("Detailed explanation"):
                        st.plotly_chart(shap_waterfall_figure(ev["shap_values"]), use_container_width=True)
                        for it in ev["shap_values"]["interpretations"]:
                            toward = "supports" if it["shap_value"] > 0 else "argues against"
                            st.markdown(f"- `{it['feature']}` = {it['actual_value']:.4g}, {toward} "
                                        f"{ev['shap_values']['explained_class']}: {it['meaning']}")
            else:
                render_note(st, "The signature check failed on a basis announcement, so the session stopped at "
                                "sifting, before any key analysis, and every key bit was destroyed.", T.ABORT)
                st.markdown('<div class="sq-reasons"><div class="ttl">Signature log</div></div>',
                            unsafe_allow_html=True)
                st.code("\n".join(out["pqc"]["log"]), language="text", wrap_lines=True)

            with st.expander("Full forensics report"):
                st.code(ev["forensics_report"], language="text")
                rp = ev.get("report_paths") or {}
                if rp.get("path_pdf") and os.path.exists(rp["path_pdf"]):
                    with open(rp["path_pdf"], "rb") as fh:
                        st.download_button("Download report as PDF", fh.read(),
                                           file_name=os.path.basename(rp["path_pdf"]))
            with st.expander("Timings and signature log"):
                st.json({k: round(v, 3) for k, v in out["timings"].items()})
                st.code("\n".join(out["pqc"]["log"]), language="text", wrap_lines=True)
                st.caption(f"Each Lamport signature is 8 KiB and each public key 16 KiB. "
                           f"Alice's key fingerprint: {out['handshake']['alice_fingerprint']}")


# --------------------------------------------------------------------------
# Tab 2 - Training results
# --------------------------------------------------------------------------
with tab_train:
    m = load_json(str(S.METRICS_PATH))
    if m is None:
        st.markdown('<div class="sq-empty"><h3>No training results yet</h3>Run '
                    '<code>python main.py train</code> to train the detector; its results appear here.</div>',
                    unsafe_allow_html=True)
    else:
        comp = m.get("comparison", {})
        if comp:
            section("Attack or normal: old rule versus ShieldQKD",
                    f"Held-out test set of {m['dataset']['n_test']:,} sessions. Higher detection and lower "
                    "missed attacks are better.")
            names = list(comp)
            pretty = {"Fixed 11% QBER": "Old 11% rule", "CUSUM": "CUSUM alarm",
                      "ML (xgboost)": "ShieldQKD (XGBoost)", "ML (random_forest)": "Random Forest",
                      "ML (svm_rbf)": "SVM"}
            xs = [pretty.get(n, n) for n in names]
            fig = go.Figure()
            fig.add_bar(x=xs, y=[comp[n]["accuracy"] * 100 for n in names], name="Detection accuracy",
                        marker_color=T.RECT, text=[f"{comp[n]['accuracy'] * 100:.1f}%" for n in names],
                        textposition="outside")
            fig.add_bar(x=xs, y=[comp[n]["fnr"] * 100 for n in names], name="Attacks missed",
                        marker_color=T.ABORT, text=[f"{comp[n]['fnr'] * 100:.1f}%" for n in names],
                        textposition="outside")
            fig.update_layout(barmode="group", height=380, yaxis=dict(range=[0, 112], ticksuffix="%"),
                              legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        section("Model scorecard",
                f"Trained on {m['dataset']['n_sessions']:,} simulated sessions with a {m['search']} search over "
                "settings. XGBoost is the model ShieldQKD uses; the other two are for comparison.")
        rows = []
        for name, e in m["models"].items():
            r = e.get("runs", {})
            rows.append({
                "Model": {"xgboost": "XGBoost", "random_forest": "Random Forest", "svm_rbf": "SVM"}.get(name, name),
                "Practice score (CV)": f"{e['cv'].get('cv_accuracy', 0) * 100:.2f}%",
                "Test accuracy": f"{e['test']['accuracy'] * 100:.2f}%",
                f"Accuracy over {r.get('n_runs', 0)} runs": (f"{r['accuracy_mean'] * 100:.2f} ± "
                                                             f"{r['accuracy_std'] * 100:.2f}%" if r else "-"),
                "Macro F1": f"{e['test']['macro_f1']:.3f}",
                "ROC-AUC": f"{e['test']['macro_roc_auc']:.3f}",
                "Attacks missed": f"{e['test']['binary']['fnr'] * 100:.2f}%",
                "Unseen test set": (f"{e['independent_test']['accuracy'] * 100:.2f}%"
                                    if e.get("independent_test") else "-"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        section("Where each model is right and wrong",
                "Confusion matrix: rows are the real situation, columns the model's answer, so the diagonal is "
                "correct. ROC curves: lines close to the top-left corner separate that class best.")
        sel = st.selectbox("Model", list(m["models"]),
                           format_func=lambda k: {"xgboost": "XGBoost", "random_forest": "Random Forest",
                                                  "svm_rbf": "SVM"}.get(k, k))
        test = m["models"][sel]["test"]
        labels = list(test["per_class_f1"])
        a, b = st.columns(2)
        with a:
            cm = np.array(test["confusion_matrix"])
            cmn = cm / cm.sum(axis=1, keepdims=True)
            fig = go.Figure(go.Heatmap(z=cmn, x=labels, y=labels, text=cm, texttemplate="%{text}",
                                       textfont=dict(color=T.TEXT),
                                       colorscale=[[0, T.PANEL], [0.12, "#283A6B"], [1, "#3D6AD0"]],
                                       zmin=0, zmax=1, showscale=False, xgap=1, ygap=1,
                                       hovertemplate="real %{y}<br>answered %{x}<br>%{text} sessions<extra></extra>"))
            fig.update_layout(title="Confusion matrix", height=470,
                              yaxis=dict(autorange="reversed", title="Real situation", showgrid=False),
                              xaxis=dict(title="Model's answer", showgrid=False))
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = go.Figure()
            for cls, curve in test["roc_curves"].items():
                fig.add_scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", line=dict(width=1.6),
                                name=f"{cls} ({test['roc_auc_per_class'][cls]:.3f})")
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=T.DIM, width=1),
                            showlegend=False, hoverinfo="skip")
            fig.update_layout(title=f"ROC curves, average AUC {test['macro_roc_auc']:.3f}", height=470,
                              xaxis_title="False alarms", yaxis_title="Attacks caught")
            st.plotly_chart(fig, use_container_width=True)
        f1 = test["per_class_f1"]
        st.dataframe(pd.DataFrame([{k: f"{v:.3f}" for k, v in f1.items()}], index=["F1 score"]),
                     use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if m.get("global_shap_importance"):
                section("What the model pays attention to",
                        "Average influence of each measurement on the decision (mean |SHAP|), top 15.")
                imp = pd.DataFrame(m["global_shap_importance"]).head(15)
                fig = go.Figure(go.Bar(x=imp["mean_abs_shap"][::-1], y=imp["feature"][::-1], orientation="h",
                                       marker_color=T.RECT))
                fig.update_layout(height=460, margin=dict(t=10), xaxis_title="Average influence")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            abl = load_json(str(S.ABLATION_PATH))
            if abl:
                full = abl["full"]["accuracy"] * 100
                section("Which groups of measurements matter",
                        f"The model retrained without each group, and with only that group. With all 63 "
                        f"measurements it scores {full:.1f}% (dashed line).")
                groups = list(abl["without"])
                nice = [g.replace("_", " ").capitalize() for g in groups]
                fig = go.Figure()
                fig.add_bar(y=nice, x=[abl["without"][g]["accuracy"] * 100 for g in groups], orientation="h",
                            name="Without this group", marker_color=T.RECT)
                fig.add_bar(y=nice, x=[abl["only"][g]["accuracy"] * 100 for g in groups], orientation="h",
                            name="Only this group", marker_color=T.DIAG)
                fig.add_vline(x=full, line_dash="dash", line_color=T.TEXT, line_width=1)
                fig.update_layout(barmode="group", height=460, margin=dict(t=10),
                                  xaxis=dict(range=[40, 100], ticksuffix="%", title="Accuracy"),
                                  yaxis=dict(autorange="reversed"),
                                  legend=dict(orientation="h", y=-0.15, x=0))
                st.plotly_chart(fig, use_container_width=True)

        base = m.get("threshold_baseline", {})
        if base.get("detection_rate_per_class"):
            section("What the old 11% rule catches",
                    "Share of each attack's test sessions that the fixed threshold flags.")
            st.dataframe(pd.DataFrame([{k: f"{v * 100:.1f}%" for k, v in base["detection_rate_per_class"].items()}],
                                      index=["Caught"]), use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3 - Forensics
# --------------------------------------------------------------------------
with tab_forensics:
    section("Session reports", "Every session writes a report. Open one to see its verdict, evidence and "
                               "recommended response.")
    render_forensics(st)
    section("What each attack triggers", "The response ShieldQKD recommends for each type of attack.")
    st.dataframe(pd.DataFrame(ResponseRouter.table(), columns=["Attack", "Recommended response"]),
                 use_container_width=True, hide_index=True)
