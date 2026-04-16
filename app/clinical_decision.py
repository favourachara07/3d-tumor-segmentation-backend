"""
Nextar NeuroVista AI — Clinical Decision Support Module
Consumes the output of calculate_clinical_volumes() from processing.py
and produces structured clinical insights.

Input contract (from processing.py → calculate_clinical_volumes):
    volume_report = {
        "volumes_mm3": {
            "total_tumor":     float,   # necrotic + edema + enhancing
            "edema":           float,   # label 2
            "necrotic_core":   float,   # label 1
            "enhancing_tumor": float,   # label 3
        },
        "derived_metrics": {
            "total_core":         float,   # necrotic + enhancing  (BraTS TC)
            "enhancing_fraction": float,   # enhancing / total_core × 100
        },
        "warnings": [...],               # QA flags from processing.py
    }

Output contract (merged back into clinical_report before JSON return):
    {
        "triage_band":             "URGENT" | "EXPEDITED" | "ROUTINE",
        "malignancy_likelihood_pct": float,   # 0-100
        "grade_hint":              str,
        "grade_probabilities": {
            "WHO_I_II": int,
            "WHO_III":  int,
            "WHO_IV":   int,
        },
        "outcome_range":  str,
        "surgical_note":  str,
        "disclaimer":     str,
    }
"""


def generate_clinical_report(volume_report: dict) -> dict:
    """
    Derive clinical decision support outputs from volumetric metrics.

    Parameters
    ----------
    volume_report : dict
        Direct output of calculate_clinical_volumes() from processing.py.

    Returns
    -------
    dict
        CDSS fields ready to be merged into the clinical_report JSON key.
    """
    vols    = volume_report["volumes_mm3"]
    derived = volume_report["derived_metrics"]

    total_tumor        = vols["total_tumor"]
    vol_enhancing      = vols["enhancing_tumor"]
    vol_necrotic       = vols["necrotic_core"]
    vol_edema          = vols["edema"]
    total_core         = derived["total_core"]          # necrotic + enhancing
    enhancing_fraction = derived["enhancing_fraction"]  # % of core that enhances

    # ── 1. Malignancy likelihood ──────────────────────────────────────────────
    # Primary driver:  enhancing_fraction (how dominant active tumour is in core)
    # Secondary driver: absolute total_core volume
    # Formula kept interpretable: weighted sum capped at 100
    frac_score   = min(enhancing_fraction * 0.65, 65.0)          # max 65 pts
    volume_score = min((total_core / 50_000) * 25.0, 25.0)       # max 25 pts  (ref: 50 cm³)
    necrotic_bonus = 10.0 if (vol_necrotic > 2_000) else 0.0     # necrotic core bonus: +10
    malignancy_score = round(min(frac_score + volume_score + necrotic_bonus, 100.0), 1)

    # ── 2. Grade classification ───────────────────────────────────────────────
    if enhancing_fraction > 60.0 or (total_tumor > 35_000 and vol_enhancing > 5_000):
        grade_hint  = "High-grade features (WHO Grade III–IV suspected)"
        grade_probs = {"WHO_I_II": 6, "WHO_III": 22, "WHO_IV": 72}

    elif enhancing_fraction > 25.0 or total_core > 10_000:
        grade_hint  = "Intermediate features (WHO Grade II–III possible)"
        grade_probs = {"WHO_I_II": 22, "WHO_III": 58, "WHO_IV": 20}

    else:
        grade_hint  = "Lower-grade features — further characterisation advised"
        grade_probs = {"WHO_I_II": 63, "WHO_III": 30, "WHO_IV": 7}

    # Sanity-check: if enhancing is essentially absent, cap WHO IV probability
    if vol_enhancing < 500:
        grade_probs = {"WHO_I_II": 55, "WHO_III": 38, "WHO_IV": 7}
        grade_hint  = "Minimal enhancement — low-grade features more likely"

    # ── 3. Triage band ────────────────────────────────────────────────────────
    # Based on absolute risk markers, not score alone.
    # URGENT:    any criterion that warrants same-day neurosurgical review
    # EXPEDITED: significant findings, review within 48 h
    # ROUTINE:   low-volume / low-grade features, elective review
    if (
        total_tumor    > 30_000    or   # > 30 cm³ total lesion
        vol_enhancing  > 8_000     or   # > 8 cm³ active tumour
        malignancy_score > 70      or   # high combined risk
        enhancing_fraction > 75.0       # dominant active core
    ):
        triage = "URGENT"

    elif (
        total_tumor    > 8_000     or
        malignancy_score > 40      or
        total_core     > 5_000
    ):
        triage = "EXPEDITED"

    else:
        triage = "ROUTINE"

    # ── 4. Outcome range (literature-grounded, conservative phrasing) ─────────
    if grade_probs["WHO_IV"] >= 60:
        outcome_range = (
            "Imaging features consistent with GBM profile (WHO IV). "
            "Median OS in comparable BraTS-profiled cases: 14–16 months. "
            "Highly dependent on MGMT methylation status, resection extent, "
            "and adjuvant chemoradiotherapy."
        )
    elif grade_probs["WHO_III"] >= 50:
        outcome_range = (
            "Features consistent with anaplastic glioma (WHO III). "
            "Median OS range: 24–48 months. "
            "IDH mutation and 1p/19q co-deletion status are critical "
            "determinants of prognosis and treatment selection."
        )
    else:
        outcome_range = (
            "Lower-grade glioma features (WHO I–II most likely). "
            "OS may exceed 5–10 years with appropriate management. "
            "Annual surveillance imaging and molecular profiling recommended."
        )

    # ── 5. Surgical note ──────────────────────────────────────────────────────
    if malignancy_score > 65:
        surgical_note = (
            "Consider supramarginal resection where neurologically safe. "
            "Awake craniotomy indicated if proximity to eloquent cortex is confirmed. "
            "Intraoperative neuromonitoring and neuronavigation strongly recommended."
        )
    elif malignancy_score > 35:
        surgical_note = (
            "Gross total resection is the primary target. "
            "Intraoperative neuromonitoring recommended. "
            "Extent of resection should be balanced against functional preservation."
        )
    else:
        surgical_note = (
            "Consider extent of resection versus functional preservation carefully. "
            "Stereotactic biopsy may be appropriate for deep-seated or eloquent lesions. "
            "Multidisciplinary neuro-oncology review recommended before intervention."
        )

    return {
        "triage_band":               triage,
        "malignancy_likelihood_pct": malignancy_score,
        "grade_hint":                grade_hint,
        "grade_probabilities":       grade_probs,
        "outcome_range":             outcome_range,
        "surgical_note":             surgical_note,
        "disclaimer": (
            "AI-assisted analysis only. All outputs require radiologist and "
            "neurosurgeon validation before clinical action. "
            "Model trained on BraTS 2020 dataset."
        ),
    }