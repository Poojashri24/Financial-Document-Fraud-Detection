from __future__ import annotations

import html
from typing import Any, Dict, List

import httpx
import streamlit as st

from frontend.integration import (
    API_BASE_URL,
    analyze_multimodal,
    analyze_tampering,
    assess_risk,
    build_consistency_graph,
    build_multimodal_document_payload,
    classify_document,
    extract_document_fields,
    format_backend_error,
    format_verification_values,
    generate_reasoning,
    process_document,
    register_document_fields,
    run_ocr,
    upload_documents,
    verify_cross_document,
)

st.set_page_config(page_title="Intelligent Financial Document Fraud Detection", page_icon="F", layout="wide")

st.markdown("""
<style>
:root { --ink:#173042; --muted:#607484; --line:#d8e1e7; --paper:#f4f7f8; --green:#176b52; --green-bg:#e8f4ef; --red:#a53737; --red-bg:#fbebea; --amber:#8c6200; --amber-bg:#fff4d7; --blue:#245d7a; }
[data-testid="stAppViewContainer"] { background:var(--paper); color:var(--ink); }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
.hero { padding:1.45rem 1.7rem; background:#173042; color:#fff; border-radius:8px; margin-bottom:1.2rem; }
.hero h1 { color:#fff; font-size:2rem; margin:0 0 .35rem 0; letter-spacing:0; }
.hero p { color:#d9e5eb; margin:0; }
.section-title { color:var(--ink); font-size:1.25rem; font-weight:700; margin:1.3rem 0 .65rem; }
.metric-card, .doc-card, .relationship, .finding-card { background:#fff; border:1px solid var(--line); border-radius:7px; padding:1rem; }
.metric-label { color:var(--muted); font-size:.82rem; }
.metric-value { color:var(--ink); font-size:1.55rem; font-weight:700; margin-top:.25rem; }
.muted { color:var(--muted); font-size:.9rem; }
.status { display:inline-block; padding:.18rem .55rem; border-radius:999px; font-size:.75rem; font-weight:700; }
.status-PASS { color:var(--green); background:var(--green-bg); }
.status-FAIL { color:var(--red); background:var(--red-bg); }
.status-WARNING { color:var(--amber); background:var(--amber-bg); }
.status-INSUFFICIENT_DATA { color:#536572; background:#e9eef1; }
.relationship { margin:.45rem 0; }
.relationship-head { display:flex; align-items:center; justify-content:space-between; gap:.65rem; flex-wrap:wrap; }
.relationship-nodes { color:var(--ink); font-weight:650; }
.relationship-label { color:var(--muted); font-size:.85rem; margin:.35rem 0; }
.field-chip { display:inline-block; background:#edf3f5; color:#274657; padding:.25rem .55rem; border-radius:4px; margin:.2rem .25rem .2rem 0; font-size:.86rem; }
.stButton>button[kind="primary"] { background:#176b52; border-color:#176b52; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:7px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Intelligent Financial Document Fraud Detection</h1>
  <p>Multimodal financial-document verification with cross-document checks and separate heuristic and trained visual tampering evidence.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### System status")
    st.caption("Multimodal financial-document verification prototype with a trained visual tampering baseline.")
    st.caption("The trained visual model provides tampering evidence and is not definitive proof of fraud.")
    st.caption("Risk score is a heuristic assessment, not a calibrated fraud probability.")
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("Backend reachable")
        else:
            st.warning(f"Backend health check returned {response.status_code}")
    except Exception as exc:
        st.error(f"Backend unavailable: {exc}")

st.markdown('<div class="section-title">Documents</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Choose financial documents",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="PDF, PNG, JPG, and JPEG files are supported.",
)
current_documents = st.session_state.get("uploaded_documents", [])
if uploaded_files:
    cards = st.columns(min(len(uploaded_files), 3))
    for index, file in enumerate(uploaded_files):
        with cards[index % len(cards)]:
            st.markdown(f"<div class='doc-card'><strong>{html.escape(file.name)}</strong><div class='muted'>{file.size / 1024:.1f} KB · Ready to upload</div></div>", unsafe_allow_html=True)

run_clicked = st.button("Run Analysis / Verify Documents", type="primary", disabled=not uploaded_files and not current_documents, use_container_width=False)

if run_clicked:
    try:
        if uploaded_files:
            with st.status("Uploading documents...", expanded=False) as status:
                upload_response = upload_documents(uploaded_files)
                current_documents = upload_response.get("documents", [])
                status.update(label=f"Uploaded {len(current_documents)} document(s)", state="complete")
            st.session_state["uploaded_documents"] = current_documents
        if not current_documents:
            st.error("Upload at least one document to begin analysis.")
        else:
            pipeline_documents: Dict[str, Dict[str, Any]] = {}
            diagnostics: Dict[str, Any] = {"upload": current_documents, "documents": {}}
            progress = st.progress(0, text="Processing documents")
            for index, document in enumerate(current_documents):
                document_id = document["document_id"]
                stage_data: Dict[str, Any] = {"document": document}
                processing = process_document(document_id)
                ocr = run_ocr(document_id)
                classification = classify_document(document_id, ocr.get("text", ""))
                fields = extract_document_fields(document_id, classification.get("document_type", "Unknown"), ocr.get("text", ""))
                fields_only = {key: value for key, value in fields.items() if key not in {"document_id", "document_type"}}
                registration = register_document_fields(document_id, classification.get("document_type", "Unknown"), fields_only)
                tampering = analyze_tampering(document_id)
                stage_data.update({"processing": processing, "ocr": ocr, "classification": classification, "fields": fields, "registration": registration, "tampering": tampering})
                diagnostics["documents"][document_id] = stage_data
                pipeline_documents[document_id] = {
                    "document_id": document_id,
                    "document_type": classification.get("document_type", "Unknown"),
                    "ocr": ocr,
                    "structured_fields": fields_only,
                    "registration": registration,
                    "tampering": tampering,
                    "metadata": {"file_name": document.get("filename"), "page_count": document.get("page_count", 1)},
                }
                progress.progress((index + 1) / len(current_documents), text=f"Processed {index + 1} of {len(current_documents)} document(s)")

            verification = verify_cross_document([item["document_id"] for item in current_documents]) if len(current_documents) > 1 else {
                "document_count": len(current_documents),
                "results": [],
                "summary": {"number_of_rules_checked": 0, "passed_rules": 0, "failed_rules": 0, "warnings": 0, "insufficient_data_checks": 0, "overall_inconsistency_count": 0},
            }
            if len(current_documents) > 1:
                result_lines = [f"{item.get('rule_name')} [{item.get('status')}]: {item.get('explanation')}" for item in verification.get("results", [])]
                overall_status = "FAIL" if verification.get("summary", {}).get("failed_rules", 0) else "WARNING" if verification.get("summary", {}).get("warnings", 0) else "PASS"
                cross_payload = {
                    "rule_id": "cross_document_verification",
                    "rule_name": "Cross-document verification",
                    "status": overall_status,
                    "documents": [item["document_id"] for item in current_documents],
                    "fields_compared": [],
                    "values": verification.get("summary", {}),
                    "explanation": "Rule-by-rule verification results are preserved in the supporting evidence.",
                    "evidence": result_lines,
                    "details": verification,
                    "gnn_evidence": verification.get("gnn_evidence", {"status": "not_configured", "predictions": []}),
                }
                for item in pipeline_documents.values():
                    item["cross_document"] = cross_payload
            diagnostics["verification"] = verification

            multimodal = analyze_multimodal([build_multimodal_document_payload(item) for item in pipeline_documents.values()])
            reasoning = generate_reasoning(analysis_id=multimodal.get("analysis_id"))
            risk = assess_risk(reasoning_id=reasoning.get("reasoning_id"))
            diagnostics.update({"multimodal": multimodal, "reasoning": reasoning, "risk": risk})
            st.session_state.update({
                "uploaded_documents": current_documents,
                "pipeline_documents": pipeline_documents,
                "verification": verification,
                "multimodal": multimodal,
                "reasoning": reasoning,
                "risk": risk,
                "diagnostics": diagnostics,
            })
            progress.progress(1.0, text="Analysis complete")
            st.success("Document analysis completed.")
    except Exception as exc:
        st.error(f"Analysis could not be completed: {format_backend_error(exc)}")

current_documents = st.session_state.get("uploaded_documents", [])
pipeline_documents = st.session_state.get("pipeline_documents", {})
verification = st.session_state.get("verification")
multimodal = st.session_state.get("multimodal")
reasoning = st.session_state.get("reasoning")
risk = st.session_state.get("risk")

if current_documents:
    st.markdown('<div class="section-title">Analysis Summary</div>', unsafe_allow_html=True)
    summary = (verification or {}).get("summary", {})
    consistency_status = "FAIL" if summary.get("failed_rules", 0) else "WARNING" if summary.get("warnings", 0) else "PASS"
    st.markdown(f"**Document Consistency:** {consistency_status} · {summary.get('overall_inconsistency_count', 0)} explicit inconsistency(ies)")
    metric_values = [
        ("Documents analyzed", (verification or {}).get("document_count", len(current_documents))),
        ("Checks passed", summary.get("passed_rules", 0)),
        ("Inconsistencies", summary.get("overall_inconsistency_count", 0)),
        ("Insufficient data", summary.get("insufficient_data_checks", 0)),
        ("Risk score", (risk or {}).get("risk_score", "—")),
        ("Risk level", (risk or {}).get("risk_level", "—")),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metric_values):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(str(value))}</div></div>", unsafe_allow_html=True)
    st.caption("Risk is a heuristic assessment, not a calibrated fraud probability or definitive fraud verdict.")
    if risk:
        st.write(risk.get("risk_explanation", ""))
        breakdown = risk.get("score_breakdown", {})
        if breakdown:
            st.markdown("**Risk evidence contributions**")
            st.table([
                {"Evidence": label, "Count": breakdown.get(key, {}).get("count", 0), "Contribution": breakdown.get(key, {}).get("contribution", breakdown.get(key, {}).get("prediction_points", 0))}
                for key, label in [
                    ("cross_document_inconsistencies", "Explicit cross-document failures"),
                    ("heuristic_tampering", "Heuristic visual observations"),
                    ("trained_visual_model", "Trained visual model (informational)"),
                    ("gnn", "GNN (advisory; no direct points)"),
                    ("multiple_suspicious_sources", "Independent suspicious sources"),
                ] if key in breakdown
            ])

    st.markdown('<div class="section-title">Cross-Document Verification</div>', unsafe_allow_html=True)
    results = (verification or {}).get("results", [])
    if not results:
        st.info("Cross-document checks require at least two uploaded documents.")
    else:
        header = st.columns([1.1, 1.3, .75, 2.2])
        for col, title in zip(header, ["Check", "Compared values", "Status", "Explanation"]):
            col.markdown(f"**{title}**")
        for result in results:
            cols = st.columns([1.1, 1.3, .75, 2.2])
            values = result.get("values") or {}
            rendered_values = format_verification_values(result.get("rule_id", ""), values)
            status = result.get("status", "INSUFFICIENT_DATA")
            color_class = status if status in {"PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"} else "INSUFFICIENT_DATA"
            cols[0].write(result.get("rule_name", "Verification check"))
            cols[1].write(rendered_values)
            cols[2].markdown(f"<span class='status status-{color_class}'>{html.escape(status.replace('_', ' '))}</span>", unsafe_allow_html=True)
            cols[3].write(result.get("explanation", "No explanation supplied."))

    st.markdown('<div class="section-title">Document Consistency Graph</div>', unsafe_allow_html=True)
    st.caption("Explainable representation of the existing rule-based checks; this is not a graph neural network.")
    graph = build_consistency_graph(verification or {}, list(pipeline_documents.values()))
    for relationship in graph["relationships"]:
        status = relationship["status"]
        status_class = status if status in {"PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"} else "INSUFFICIENT_DATA"
        left = html.escape(relationship["left"].replace("\n", " · "))
        right = html.escape(relationship["right"].replace("\n", " · "))
        st.markdown(f"<div class='relationship'><div class='relationship-head'><span class='relationship-nodes'>{left} ↔ {right}</span><span class='status status-{status_class}'>{html.escape(status.replace('_', ' '))}</span></div><div class='relationship-label'>{html.escape(relationship['label'])}</div></div>", unsafe_allow_html=True)
        if relationship["values"]:
            values = format_verification_values(relationship.get("rule_id", ""), relationship["values"])
            st.markdown(f"<div class='muted'>{html.escape(values)}</div>", unsafe_allow_html=True)
        st.caption(relationship["explanation"])

    gnn_result = (verification or {}).get("gnn_evidence", {})
    st.markdown("**Graph-Based Cross-Document Analysis**")
    st.caption("Learned graph-based evidence distilled from deterministic verification scenarios; advisory only and not proof of fraud.")
    if gnn_result.get("predictions"):
        for prediction in gnn_result["predictions"]:
            rule = next((item for item in results if item.get("rule_id") == prediction.get("rule_id")), {})
            left_id, right_id = (prediction.get("documents") or ["Document", "Document"])[:2]
            left_name = next((doc.get("filename", left_id) for doc in current_documents if doc.get("document_id") == left_id), left_id)
            right_name = next((doc.get("filename", right_id) for doc in current_documents if doc.get("document_id") == right_id), right_id)
            st.markdown(f"**{left_name} ↔ {right_name}**")
            st.write(f"Rule-based result: {prediction.get('rule_based_status', rule.get('status', 'UNKNOWN'))} · GNN consistency prediction: {prediction.get('gnn_prediction', 'UNKNOWN')} · Confidence: {prediction.get('confidence', 0.0):.3f}")
            st.caption(prediction.get("explanation", "GNN is advisory and cannot override the explicit rule result."))
    else:
        st.caption(f"Graph model status: {gnn_result.get('status', 'not available')}. Rule-based verification remains authoritative.")

    st.markdown('<div class="section-title">Document Findings</div>', unsafe_allow_html=True)
    for document in current_documents:
        document_id = document["document_id"]
        item = pipeline_documents.get(document_id, {})
        with st.expander(f"{document.get('filename', 'Document')} · {item.get('document_type', 'Type unavailable')}", expanded=False):
            st.write(f"Processing: {item.get('processing', {}).get('status', 'Complete')} · Pages: {item.get('processing', {}).get('pages_processed', document.get('page_count', 1))}")
            st.markdown("**Extracted fields**")
            fields = item.get("structured_fields", {})
            if fields:
                st.table([{"Field": key.replace("_", " ").title(), "Value": value if value is not None else "Not extracted"} for key, value in fields.items() if value not in (None, "", [])])
            else:
                st.caption("No structured fields were extracted.")
            st.markdown("**OCR details**")
            ocr = item.get("ocr", {})
            st.caption(f"OCR status: {ocr.get('status', 'unavailable')} · Pages: {ocr.get('pages', document.get('page_count', 1))}")
            text = ocr.get("text", "")
            st.text(text[:1600] + ("\n…" if len(text) > 1600 else "") if text else "No OCR text available.")
            tampering = item.get("tampering", {})
            st.markdown("**Forensic heuristic**")
            st.write(f"Suspicious signal: {'Yes' if tampering.get('tampering_detected') else 'No'}")
            for finding in tampering.get("findings", []):
                st.markdown(f"- **{finding.get('finding', 'Finding')}** ({finding.get('severity', 'unspecified')}): {finding.get('explanation', '')}")
                if finding.get("region"):
                    st.caption(f"Region: {finding['region']}")
            st.caption("Heuristic observations indicate suspicious characteristics, not confirmed tampering or fraud.")
            ml = tampering.get("ml_evidence", {})
            st.markdown("**Trained visual model evidence**")
            if ml.get("prediction"):
                st.write(f"Prediction: {ml['prediction'].title()} · Forged probability: {ml.get('forged_probability', 'n/a')}")
                st.caption(f"{ml.get('model_name', 'Trained visual model')} · Model output is not proof of fraud.")
            else:
                st.caption(ml.get("reason", "Model evidence unavailable."))
            st.markdown("**Evidence limitations**")
            for limitation in tampering.get("limitations", []):
                st.caption(f"- {limitation}")
            if ml.get("calibration_note"):
                st.caption(f"- {ml['calibration_note']}")

    st.markdown('<div class="section-title">Explainable Verification Report</div>', unsafe_allow_html=True)
    failed = [item for item in results if item.get("status") in {"FAIL", "WARNING"}]
    if failed:
        st.warning(f"{len(failed)} verification check(s) need review.")
    elif results:
        st.success("No cross-document FAIL or WARNING results were returned.")
    st.markdown("**Cross-document findings**")
    for result in failed:
        st.markdown(f"- **{result.get('rule_name')}**: {result.get('explanation')}")
    st.markdown("**Reasoning mode**")
    st.write((reasoning or {}).get("reasoning_mode", "Not available"))
    if reasoning and reasoning.get("reasoning_mode") == "heuristic_rule_based":
        st.caption("Deterministic evidence-grounded reasoning is active; no external LLM is implied.")
    st.markdown("**Visual observations**")
    for document in current_documents:
        record = pipeline_documents.get(document["document_id"], {})
        tampering = record.get("tampering", {})
        ml = tampering.get("ml_evidence", {})
        for finding in tampering.get("findings", []):
            st.markdown(f"- {document.get('filename')}: heuristic {finding.get('severity', 'signal')} — {finding.get('explanation', finding.get('finding', ''))}")
        if ml.get("prediction"):
            st.markdown(f"- {document.get('filename')}: trained visual model predicted **{ml['prediction']}** (forged probability {ml.get('forged_probability', 'n/a')}); model evidence only.")
    st.markdown("**Manual review and limitations**")
    limitations = list((reasoning or {}).get("limitations", [])) + list((risk or {}).get("limitations", []))
    for document in current_documents:
        limitations.extend(pipeline_documents.get(document["document_id"], {}).get("tampering", {}).get("limitations", []))
    for limitation in dict.fromkeys(limitations):
        st.write(f"- {limitation}")
    if not limitations:
        st.write("- Automated outputs are evidence for review, not a definitive fraud verdict.")

with st.expander("Developer Diagnostics", expanded=False):
    diagnostics = st.session_state.get("diagnostics")
    if diagnostics:
        st.json(diagnostics)
    else:
        st.caption("Raw API responses will appear here after an analysis run.")
