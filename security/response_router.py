"""Attack class -> countermeasure routing (spec Section 8.2)."""
from __future__ import annotations

RESPONSES = {
    "Normal": {
        "action": "Proceed: run error correction + privacy amplification and use the key",
        "severity": "none", "abort": False, "switch_protocol": None,
        "details": ["Key accepted", "Continue routine temporal-QBER monitoring"],
    },
    "Classical MITM": {
        "action": "Abort: classical channel authentication failed - destroy all key material",
        "severity": "critical", "abort": True, "switch_protocol": None,
        "details": ["Discard sifted key and every derived secret",
                    "Re-run the PQC handshake over a trusted out-of-band channel",
                    "Investigate the classical network path (possible active MITM)"],
    },
    "Intercept-Resend": {
        "action": "Abort + switch to Ekert91 protocol",
        "severity": "high", "abort": True, "switch_protocol": "Ekert91",
        "details": ["Discard this key", "Re-establish with Ekert91: the Bell-CHSH test exposes "
                    "any measure-and-resend adversary (S drops to <= 2)",
                    "Inspect the fibre route for an inline measurement device"],
    },
    "PNS": {
        "action": "Enable decoy-state, reduce mean photon number",
        "severity": "high", "abort": True, "switch_protocol": None,
        "details": ["Add decoy states to the source (signal + decoy intensities)",
                    "Lower mean photon number mu to suppress multi-photon pulses",
                    "Compare yields of decoy vs signal pulses to bound Eve's information"],
    },
    "Beam-Splitting": {
        "action": "Increase privacy amplification compression",
        "severity": "medium", "abort": False, "switch_protocol": None,
        "details": ["Shrink the final key by the loss-derived leakage bound",
                    "Monitor transmittance against the calibrated channel loss"],
    },
    "Time-Shift": {
        "action": "Randomize detector timing gates",
        "severity": "high", "abort": True, "switch_protocol": None,
        "details": ["Randomise the detection-gate position / width per time slot",
                    "Check detector efficiency mismatch between bases",
                    "Discard rounds detected near the gate edges"],
    },
    "Trojan-Horse": {
        "action": "Activate optical isolator + abort",
        "severity": "critical", "abort": True, "switch_protocol": None,
        "details": ["Engage optical isolators / filters on Alice's and Bob's ports",
                    "Monitor back-reflected light with a watchdog detector",
                    "Discard the key - Eve may have read basis settings directly"],
    },
    "Quantum-Cloning": {
        "action": "Increase sifting sample for tighter QBER",
        "severity": "high", "abort": True, "switch_protocol": None,
        "details": ["Enlarge the parameter-estimation sample to tighten the QBER bound",
                    "QBER close to the cloning limit leaves no secret key - abort"],
    },
    "Fake-State": {
        "action": "Cross-validate with Bell-CHSH test (S > 2)",
        "severity": "high", "abort": True, "switch_protocol": "Ekert91",
        "details": ["Run an entanglement-based round and verify CHSH violation S > 2",
                    "Check rectilinear vs diagonal error asymmetry and detector mismatch"],
    },
}


class ResponseRouter:
    @staticmethod
    def get_action(attack_class: str) -> str:
        return RESPONSES.get(attack_class, RESPONSES["Normal"])["action"]

    @staticmethod
    def get_response(attack_class: str) -> dict:
        r = dict(RESPONSES.get(attack_class, RESPONSES["Normal"]))
        r["attack_class"] = attack_class
        return r

    @staticmethod
    def table() -> list[tuple[str, str]]:
        return [(k, v["action"]) for k, v in RESPONSES.items()
                if k not in ("Normal", "Classical MITM")]
