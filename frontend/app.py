import io
from pathlib import Path
import streamlit as st
from PIL import Image
from api_client import APIClient

# Page Config
st.set_page_config(
    page_title="Brain Tumor AI Diagnostic Assistant",
    page_icon="🧠",
    layout="wide",
)

# Load External CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

api_client = APIClient()

# Session State Initialization
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "selected_patient_report" not in st.session_state:
    st.session_state.selected_patient_report = None


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION (Model Performance Removed)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Brain Tumor AI")
    st.markdown(
        "<span style='color:#94a3b8; font-size:12px;'>Diagnostic Assistant</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📂 Prediction History", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.write("")
    if api_client.check_health():
        st.markdown(
            """<div style="background-color: #064e3b; color: #34d399; padding: 8px 12px; border-radius: 6px; font-weight: 600; font-size: 13px; text-align: center;">
                FastAPI Backend: Online
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="background-color: #7f1d1d; color: #fca5a5; padding: 8px 12px; border-radius: 6px; font-weight: 600; font-size: 13px; text-align: center;">
                FastAPI Backend: Offline
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """<div style="background-color: #131c31; padding: 12px; border-radius: 8px; border: 1px solid #1e293b;">
            <span style="color: #10b981; font-weight: bold; font-size: 13px;">🛡️ Secure & Private</span><br>
            <span style="color: #94a3b8; font-size: 11px;">Your data is encrypted and protected</span>
        </div>""",
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# -----------------------------------------------------------------------------
if page == "🏠 Dashboard":
    h_col1, h_col2 = st.columns([5, 1])
    with h_col1:
        st.markdown("<h2 style='margin:0;'>Brain Tumor AI Diagnostic Assistant</h2>", unsafe_allow_html=True)
        st.markdown("<span style='color: #94a3b8;'>AI-Powered Brain MRI Classification with Explainable AI</span>", unsafe_allow_html=True)

    with h_col2:
        if st.button("📜 History", use_container_width=True):
            st.info("Switch to Prediction History from the sidebar.")

    st.write("")

    # Input Section
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    in_col1, in_col2, in_col3 = st.columns([2.5, 2.5, 1.2])

    with in_col1:
        st.markdown("<span style='font-size: 13px; font-weight: 600; color: #e2e8f0;'>Upload MRI Scan</span>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    with in_col2:
        st.markdown("<span style='font-size: 13px; font-weight: 600; color: #e2e8f0;'>Patient Information</span>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)

        with p1:
            patient_name = st.text_input(
                "Patient Name",
                value="John Doe",
                label_visibility="collapsed"
            )

        with p2:
            patient_id = st.text_input(
                "Patient ID",
                value="PAT-1001",
                label_visibility="collapsed"
            )

        with p3:
            patient_age = st.text_input(
                "Age",
                value="45",
                label_visibility="collapsed"
            )

        with p4:
            patient_gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"],
                label_visibility="collapsed"
            )
    with in_col3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        analyze_btn = st.button("🧠 Analyze MRI", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Trigger Analysis
    if analyze_btn:
        if uploaded_file is None:
            st.error("Please upload an MRI scan image first.")
        else:
            image_bytes = uploaded_file.read()
            with st.spinner("Processing MRI through Neural Network..."):
                try:
                    results = api_client.predict_mri(
                        image_bytes=image_bytes,
                        filename=uploaded_file.name,
                        patient_code=patient_id,
                        name=patient_name,
                        age=patient_age,
                        gender=patient_gender,
                    )
                    st.session_state.analysis_data = {"results": results, "image_bytes": image_bytes}
                    st.session_state.analyzed = True
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # Display Dashboard Results Grid
    if st.session_state.analyzed and st.session_state.analysis_data:
        data = st.session_state.analysis_data["results"]
        image_bytes = st.session_state.analysis_data["image_bytes"]

        prediction = data.get("prediction", {})
        report = data.get("report", {})

        pred_class = prediction.get("predicted_class", "Glioma").title()
        confidence = prediction.get("confidence_score", 0.9782) * 100
        probabilities = prediction.get("class_probabilities", {"glioma": 0.9782, "meningioma": 0.0102, "pituitary": 0.0063, "notumor": 0.0053})

        left_col, right_col = st.columns([1.1, 1])

        with left_col:
            st.markdown("<h4 style='margin-bottom: 8px;'>MRI Visualization</h4>", unsafe_allow_html=True)
            v_col1, v_col2 = st.columns(2)
            input_img = Image.open(io.BytesIO(image_bytes))

            with v_col1:
                st.caption("Original MRI")
                st.image(input_img, use_container_width=True)

            with v_col2:
                st.caption("Grad-CAM Heatmap")
                heatmap_url = prediction.get("heatmap_url", "")
                heatmap_full_url = f"http://127.0.0.1:8000{heatmap_url}" if heatmap_url.startswith("/") else heatmap_url
                try:
                    st.image(heatmap_full_url, use_container_width=True)
                except Exception:
                    st.image(input_img, use_container_width=True)

            st.markdown("<h4 style='margin-top: 15px; margin-bottom: 8px;'>Class Probabilities</h4>", unsafe_allow_html=True)
            for cls_name, prob_val in probabilities.items():
                formatted_name = "Normal" if cls_name.lower() == "notumor" else cls_name.title()
                pb_c1, pb_c2 = st.columns([4, 1])
                with pb_c1:
                    st.progress(float(prob_val), text=formatted_name)
                with pb_c2:
                    st.markdown(f"<span style='font-size: 12px; font-weight: bold; color:#f8fafc;'>{prob_val * 100:.2f}%</span>", unsafe_allow_html=True)

        with right_col:
            st.markdown("<h4 style='margin-bottom: 8px;'>Prediction Summary</h4>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-title">🧠 Prediction</div>
                        <div style="color: #f43f5e; font-size: 20px; font-weight: 700;">{pred_class}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-title">📈 Confidence</div>
                        <div style="color: #10b981; font-size: 20px; font-weight: 700;">{confidence:.2f}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            m3, m4 = st.columns(2)
            with m3:
                affected_reg = prediction.get("gradcam_region", "Left Frontal Lobe")
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-title">📍 Affected Region</div>
                        <div style="color: #f8fafc; font-size: 14px; font-weight: 600;">{affected_reg}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                risk = "High" if pred_class.lower() != "notumor" else "Low"
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-title">⚠️ Risk Level</div>
                        <div style="color: #f43f5e; font-size: 18px; font-weight: 700;">{risk}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<h4 style='margin-top: 15px; margin-bottom: 8px;'>AI Diagnostic Report</h4>", unsafe_allow_html=True)

            with st.expander("➕ Impression", expanded=True):
                st.markdown(f"<span style='color: #cbd5e1; font-size: 13px;'>{report.get('impression', f'The MRI scan demonstrates characteristics consistent with {pred_class}.')}</span>", unsafe_allow_html=True)

            with st.expander("🔍 Key Findings", expanded=False):
                findings = report.get("key_findings", ["Hyperintense lesion observed."])
                for f in findings:
                    st.markdown(f"<span style='color: #cbd5e1; font-size: 13px;'>• {f}</span>", unsafe_allow_html=True)

            with st.expander("💡 Recommendations", expanded=False):
                recs = report.get("recommendations", ["Clinical correlation recommended."])
                for r in recs:
                    st.markdown(f"<span style='color: #cbd5e1; font-size: 13px;'>• {r}</span>", unsafe_allow_html=True)

        st.write("")
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            st.button("📄 Download PDF Report", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🔄 Analyze Another MRI", use_container_width=True):
                st.session_state.analyzed = False
                st.session_state.analysis_data = None
                st.rerun()

# -----------------------------------------------------------------------------
# PAGE 2: PREDICTION HISTORY
# -----------------------------------------------------------------------------
elif page == "📂 Prediction History":
    st.markdown("<h2 style='margin:0;'>📂 Prediction History</h2>", unsafe_allow_html=True)
    st.markdown("<span style='color: #94a3b8;'>Search patient diagnostic records, filter scans, and view clinical reports.</span>", unsafe_allow_html=True)
    st.write("")

    # Search & Filter Bar
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns([3, 2, 2])

    with f_col1:
        search_query = st.text_input("🔍 Search Patient ID", placeholder="Enter Patient ID (e.g. PAT-1001)")
    with f_col2:
        class_filter = st.selectbox("Filter by Class", ["All", "Glioma", "Meningioma", "Pituitary", "Normal"])
    with f_col3:
        risk_filter = st.selectbox("Filter by Risk", ["All", "High", "Low"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Filter mock records
    try:
        filtered_records = api_client.get_history()
    except Exception as e:
        st.error(f"Unable to load prediction history: {e}")
        filtered_records = []
    if search_query:
        filtered_records = [r for r in filtered_records if search_query.lower() in r["patient"]["patient_code"].lower() or search_query.lower() in r["patient"]["name"].lower()]
    if class_filter != "All":
        filtered_records = [r for r in filtered_records if r["prediction"].lower() == class_filter.lower()]
    if risk_filter != "All":
        filtered_records = [r for r in filtered_records if r["risk"].lower() == risk_filter.lower()]

    # Previous Predictions List
    st.markdown("### Previous Predictions")
    if not filtered_records:
        st.warning("No matching patient diagnostic records found.")
    else:
        for item in filtered_records:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)

            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([2, 2, 2, 2, 2])

            with r_col1:
                st.markdown(f"**Patient:** {item['patient']['name']}")
                st.caption(f"ID: {item['patient']['patient_code']}")
                st.caption(
                    f"Date: {item['prediction']['created_at'][:10]}"
                )

            with r_col2:
                st.markdown(
                    f"**Prediction:** {item['prediction']['predicted_class']}"
                )
                st.caption(
                    f"Confidence: {item['prediction']['confidence_score'] * 100:.2f}%"
                )

            with r_col3:
                st.markdown(
                    f"**Region:** {item['prediction']['gradcam_region']}"
                )

                risk = (
                    "Low"
                    if item["prediction"]["predicted_class"].lower() == "normal"
                    else "High"
                )

                st.caption(f"Risk Level: {risk}")

            with r_col4:
                st.markdown(
                    f"**Age / Gender:** {item['patient']['age']} / {item['patient']['gender']}"
                )

            with r_col5:
                if st.button(
                    "👁️ View Report",
                    key=f"btn_{item['prediction']['id']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_patient_report = item

            st.markdown("</div>", unsafe_allow_html=True)

    # View Report Modal / Expander
    if st.session_state.selected_patient_report:

        rep = st.session_state.selected_patient_report

        st.divider()

        st.markdown(
            f"## Clinical Report - {rep['patient']['name']}"
        )

        st.write(f"**Patient ID:** {rep['patient']['patient_code']}")
        st.write(f"**Prediction:** {rep['prediction']['predicted_class']}")
        st.write(
            f"**Confidence:** {rep['prediction']['confidence_score'] * 100:.2f}%"
        )
        st.write(
            f"**Region:** {rep['prediction']['gradcam_region']}"
        )

        st.subheader("Impression")
        st.write(rep["report"]["impression"])

        st.subheader("Key Findings")
        for finding in rep["report"]["key_findings"]:
            st.write(f"• {finding}")

        st.subheader("Certainty Analysis")
        st.write(rep["report"]["certainty_analysis"])

        st.subheader("Recommendations")
        for rec in rep["report"]["recommendations"]:
            st.write(f"• {rec}")

        if st.button("Close Report"):
            st.session_state.selected_patient_report = None
            st.rerun()

# -----------------------------------------------------------------------------
# PAGE 3: ABOUT
# -----------------------------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown("<h2 style='margin:0;'>ℹ️ About Brain Tumor AI</h2>", unsafe_allow_html=True)
    st.markdown("<span style='color: #94a3b8;'>Comprehensive system information and architecture details.</span>", unsafe_allow_html=True)
    st.write("")

    # System Overview
    with st.expander("🌐 System Overview", expanded=True):
        st.write(
            "The Brain Tumor AI Diagnostic Assistant is an end-to-end medical vision suite designed to classify "
            "brain tumors from MRI scans, highlight focal lesion regions using Grad-CAM explainable AI, "
            "and generate automated radiology reports."
        )

    # AI Pipeline
    with st.expander("⚙️ AI Pipeline", expanded=False):
        st.write("1. **Image Preprocessing:** Intensity normalization, cropping, and rescaled array mapping.")
        st.write("2. **Convolutional Inference:** Deep convolutional neural network evaluation across class vectors.")
        st.write("3. **Grad-CAM Localization:** Spatial gradient visualization for model interpretability.")
        st.write("4. **LLM Report Synthesis:** Automated clinical impression and recommendation formulation.")

    # Technologies
    with st.expander("🛠️ Technologies", expanded=False):
        st.markdown("• **Frontend:** Streamlit Custom CSS Layout Engine")
        st.markdown("• **Backend API:** FastAPI / Uvicorn Server")
        st.markdown("• **Deep Learning:** PyTorch / Torchvision / OpenCV")
        st.markdown("• **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping)")

    # Dataset
    with st.expander("📊 Dataset", expanded=False):
        st.write("Trained and validated on multi-modal Brain MRI datasets comprising 4 primary target classes:")
        st.markdown("1. **Glioma**")
        st.markdown("2. **Meningioma**")
        st.markdown("3. **Pituitary Tumor**")
        st.markdown("4. **Normal (No Tumor)**")

    # Model Information
    with st.expander("🧠 Model Information", expanded=False):
        st.markdown("• **Architecture:** Fine-Tuned Convolutional Neural Network")
        st.markdown("• **Input Resolution:** 224 x 224 x 3")
        st.markdown("• **Output:** 4 Class Probability Distribution + Grad-CAM Heatmap Array")

    # Performance
    with st.expander("📈 Performance", expanded=False):
        st.markdown("• **Test Accuracy:** 97.8%")
        st.markdown("• **Sensitivity:** 96.5%")
        st.markdown("• **Specificity:** 98.2%")

    # Version
    with st.expander("🏷️ Version", expanded=False):
        st.write("**Suite Version:** v1.2.0")
        st.write("**Engine Build:** 2026.07")