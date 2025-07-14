import streamlit as st
import requests

st.set_page_config(page_title="Pharma Strategy Agent", layout="wide")
st.title("Snap Strategy System")

st.markdown("Select filters for your company's portfolio below:")

# === FILTERS ===
col1, col2, col3 = st.columns(3)

with col1:
    for_sale = st.selectbox("For Sale Status", ["all", "for_sale", "for_purchase", "sold"])
    asset_type = st.selectbox("Asset Type", ["all", "molecule", "platform", "device"])
    min_deal = st.number_input("Min Deal Value ($M)", min_value=0, value=5)
    max_deal = st.number_input("Max Deal Value ($M)", min_value=0, value=100)

with col2:
    stage_dev = st.selectbox("Stage of Development", ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Marketed"])
    therapeutic_area = st.text_input("Therapeutic Area", "Oncology")
    indication = st.text_input("Indication", "Lung Cancer")
    moa = st.text_input("Mechanism of Action", "EGFR Inhibitor")
    route = st.text_input("Route of Administration", "")
    modality = st.text_input("Modality", "Small Molecule")
    patent_expiry = st.number_input("Patent Expiry Year", min_value=2024, step=1, format="%d", value=2030)

with col3:
    company_type = st.selectbox("Company Type", ["Biotech", "Pharma", "Academic", "Startup"])
    dev_stage = st.selectbox("Company Development Stage", ["Preclinical", "Clinical Stage", "Commercial"])
    hq = st.text_input("Headquarters", "USA")
    financial_status = st.selectbox("Financial Status", ["Stable", "Struggling", "Growing"])
    partner_status = st.selectbox("Partner Status", ["Partnered", "Unpartnered"])
    territory = st.text_input("Territory", "")

st.markdown("### Peak Sales Estimations")
col4, col5, col6 = st.columns(3)
with col4:
    sales_1yr = st.number_input("1-Year Sales Potential ($M)", min_value=0, value=25)
with col5:
    sales_5yr = st.number_input("5-Year Sales Potential ($M)", min_value=0, value=0)
with col6:
    peak_sales = st.number_input("Peak Sales ($M)", min_value=0, value=0)

st.markdown("### Strategic Objective")
strategic_objective = st.selectbox(
    "Choose Objective",
    ["Pruning the Portfolio", "Diversification", "Filling the Pipeline", "Augmenting the Pipeline", "Custom Query"]
)
custom_query = ""
if strategic_objective == "Custom Query":
    custom_query = st.text_area("Enter Custom Query")

# Submit button
if st.button(" Generate Strategy Recommendations"):
    # === Build input JSON ===
    input_json = {
        "filters": {
            "for_sale": for_sale,
            "asset_type": asset_type,
            "deal_value": {
                "min": min_deal,
                "max": max_deal
            },
            "asset_phenotype": {
                "stages_of_development": stage_dev,
                "therapeutic_area": therapeutic_area,
                "indication": indication,
                "mechanism_of_action": moa,
                "route_of_administration": route,
                "modality": modality,
                "patent_expiry_yr": patent_expiry
            },
            "company_details": {
                "company_type": company_type,
                "development_stage": dev_stage,
                "headquarters": hq,
                "financial_status": financial_status,
                "partner_status": partner_status,
                "territory": territory
            },
            "peak_sales": {
                "1yr_sales_potential": sales_1yr,
                "5yr_sales_potential": sales_5yr or None,
                "peak_sales": peak_sales or None
            },
            "strategic_objective": strategic_objective,
            "custom_query_text": custom_query
        },
        "current_company_id": "your_company_id_123"
    }

    try:
        # Call Flask API
        with st.spinner("Generating strategy..."):
            response = requests.post("http://127.0.0.1:5000/generate-strategy", json=input_json)
            data = response.json()

        if "recommendations" in data:
            st.success("Strategy Recommendations Received")
            st.json(data["recommendations"])
        # if "recommendations" in data:
        #     st.success("Strategy Recommendations Received")
        #     for i, rec in enumerate(data["recommendations"], 1):
        #         st.markdown(f"**{i}. Filter:** `{rec['filter']}`")
        #         st.markdown(f"- Suggested Change: `{rec['suggested_change']}`")
        #         st.markdown(f"- Reason: {rec['reason']}")
        else:
            st.error(f"API Error: {data.get('error', 'Unknown error')}")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
