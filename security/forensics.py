"""
Attack forensics reports (.json + .txt, optional .pdf) written to reports/.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np

from config import settings as S


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 512 else f"<array len={o.size}>"
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def clean_metadata(meta: dict) -> dict:
    """Drop raw per-qubit arrays from session metadata."""
    return {k: _jsonable(v) for k, v in (meta or {}).items()
            if k not in ("matching_indices", "lost_mask")}


class Forensics:
    @staticmethod
    def generate(*, decision: str, attack_class: str | None, confidence: float,
                 probabilities: dict | None, shap_values: dict | None, qber_stats: dict | None,
                 response: dict, session_metadata: dict | None, pqc_verified: bool,
                 pqc_log: list | None = None, threshold_verdict: dict | None = None,
                 qber_sequence=None, save: bool = True, out_dir: str = str(S.REPORTS_DIR),
                 pdf: bool = False) -> dict:
        ts = datetime.now(timezone.utc)
        report_id = f"SQKD-{ts.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        meta = clean_metadata(session_metadata)
        top = (shap_values or {}).get("top_features", [])
        data = {
            "report_id": report_id,
            "timestamp_utc": ts.isoformat(),
            "decision": decision,
            "attack_class": attack_class,
            "confidence": float(confidence),
            "pqc_verified": bool(pqc_verified),
            "pqc_log": pqc_log or [],
            "all_probabilities": _jsonable(probabilities or {}),
            "shap": {
                "explained_class": (shap_values or {}).get("explained_class"),
                "base_value": (shap_values or {}).get("base_value"),
                "method": (shap_values or {}).get("method"),
                "top_features": [{"feature": n, "shap_value": float(v), "actual_value": float(a)}
                                 for n, v, a in top],
                "interpretations": _jsonable((shap_values or {}).get("interpretations", [])),
            } if shap_values else None,
            "qber_stats": _jsonable(qber_stats or {}),
            "fixed_threshold_verdict": _jsonable(threshold_verdict or {}),
            "recommended_action": response.get("action"),
            "response": _jsonable(response),
            "session": meta,
            "qber_sequence": _jsonable(np.asarray(qber_sequence)) if qber_sequence is not None else None,
        }
        text = Forensics.format_text(data)
        data["text"] = text
        if save:
            os.makedirs(out_dir, exist_ok=True)
            jp = os.path.join(out_dir, f"{report_id}.json")
            tp = os.path.join(out_dir, f"{report_id}.txt")
            with open(jp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            with open(tp, "w", encoding="utf-8") as fh:
                fh.write(text)
            data["path_json"], data["path_txt"] = jp, tp
            if pdf:
                data["path_pdf"] = Forensics.save_pdf(data, os.path.join(out_dir, f"{report_id}.pdf"))
        return data

    @staticmethod
    def format_text(d: dict) -> str:
        line = "=" * 72
        out = [line, "  SHIELDQKD SECURITY FORENSICS REPORT", line,
               f"Report ID      : {d['report_id']}",
               f"Timestamp (UTC): {d['timestamp_utc']}",
               f"DECISION       : {d['decision']}",
               f"Attack class   : {d['attack_class'] or 'None'}",
               f"Confidence     : {d['confidence'] * 100:.1f}%",
               f"PQC signature  : {'VERIFIED' if d['pqc_verified'] else 'FAILED'} (Lamport OTS / SHA-256)",
               ""]
        s = d.get("session") or {}
        if s:
            out += ["-- Session " + "-" * 61,
                    f"Protocol / backend : {s.get('protocol')} / {s.get('backend')}",
                    f"Qubits sent        : {s.get('n_qubits')}   sifted: {s.get('n_sifted')}",
                    f"Overall QBER       : {100 * (s.get('overall_qber') or 0):.2f}%",
                    f"Photon loss rate   : {100 * (s.get('photon_loss_rate') or 0):.2f}%",
                    f"Noise              : {s.get('noise_model')} p={s.get('noise_prob')}",
                    ""]
        q = d.get("qber_stats") or {}
        if q:
            out += ["-- Temporal QBER " + "-" * 55]
            out += [f"{k:22s}: {v:.4f}" if isinstance(v, float) else f"{k:22s}: {v}" for k, v in q.items()]
            out.append("")
        tv = d.get("fixed_threshold_verdict") or {}
        if tv:
            out += ["-- Conventional 11% threshold " + "-" * 42,
                    f"Verdict: {tv.get('verdict')}  (mean QBER {100 * tv.get('mean_qber', 0):.2f}%)",
                    ""]
        probs = d.get("all_probabilities") or {}
        if probs:
            out.append("-- Classifier probabilities " + "-" * 44)
            for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
                out.append(f"{k:18s} {v * 100:6.2f}%  {'#' * int(round(v * 40))}")
            out.append("")
        sh = d.get("shap")
        if sh and sh.get("top_features"):
            out.append(f"-- SHAP explanation (class: {sh.get('explained_class')}) " + "-" * 20)
            for it in sh.get("interpretations") or []:
                out.append(f"{it['feature']:24s} shap={it['shap_value']:+.3f} value={it['actual_value']:.4f}"
                           f"  {it['direction']} this class - {it['meaning']}")
            out.append("")
        r = d.get("response") or {}
        out += ["-- Recommended response " + "-" * 48,
                f"Action  : {r.get('action')}",
                f"Severity: {r.get('severity')}"]
        out += [f"  * {x}" for x in r.get("details", [])]
        if d.get("pqc_log"):
            out += ["", "-- PQC audit log " + "-" * 55] + [f"  {x}" for x in d["pqc_log"]]
        out.append(line)
        return "\n".join(out)

    @staticmethod
    def save_pdf(d: dict, path: str) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.97, d["text"], family="monospace", fontsize=6.2, va="top")
            pdf.savefig(fig)
            plt.close(fig)
            fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
            seq = d.get("qber_sequence")
            if isinstance(seq, list) and seq:
                axes[0].plot(np.arange(1, len(seq) + 1), seq, marker="o", ms=3)
                axes[0].axhline(S.FIXED_QBER_THRESHOLD, color="r", ls="--", label="11% threshold")
                axes[0].set_xlabel("window")
                axes[0].set_ylabel("QBER")
                axes[0].set_title("Temporal QBER")
                axes[0].legend()
            sh = d.get("shap") or {}
            tops = sh.get("top_features") or []
            if tops:
                names = [t["feature"] for t in tops][::-1]
                vals = [t["shap_value"] for t in tops][::-1]
                axes[1].barh(names, vals, color=["#d1495b" if v > 0 else "#2b7bba" for v in vals])
                axes[1].set_title(f"Top SHAP contributions ({sh.get('explained_class')})")
                axes[1].set_xlabel("SHAP value (log-odds)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        return path


def list_reports(out_dir: str = str(S.REPORTS_DIR)) -> list[str]:
    if not os.path.isdir(out_dir):
        return []
    files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".json")]
    return sorted(files, key=os.path.getmtime, reverse=True)
