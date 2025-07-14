import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Gemini API Key and Model ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyC4q-ry8oPTjBHDP1suYrtB2PX52MXREwg"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
parser = JsonOutputParser()

# ------------------ PROMPT TEMPLATES ------------------

BASE_PROMPT_TEMPLATE = """
You are a pharma strategy AI assistant.

Objective: {strategic_objective}

Compare the current company’s molecule portfolio with competitor portfolios.
{strategy_details}

Only suggest meaningful, evidence-backed changes. Return JSON:
[
  {{
    "filter": "<filter_path>",
    "suggested_change": "<new_value>",
    "reason": "<why this should be changed>"
  }}
]

CURRENT COMPANY FILTERS:
{user_filters}

COMPETITOR DATA:
{competitor_data}
"""

STRATEGY_DETAILS = {
    "Pruning the Portfolio": """
Goal: Identify and recommend assets for de-prioritization from the current portfolio.

Focus: Molecules stuck in early development, low ROI, clinical failure, poor market performance, expiring patents.
Key factors: development stage, competitor saturation, risk scores (toxicity, ADMET), market launch failure.
""",
    "Diversification": """
Goal: Recommend molecules to broaden the portfolio across new indications, MoAs, or modalities.

Focus: New therapeutic areas, orthogonal mechanisms, different administration routes, unmet market needs.
Key factors: MeSH gaps, modality diversity, orphan drug opportunities.
""",
    "Filling the Pipeline": """
Goal: Recommend molecules to strengthen and balance the development pipeline.

Focus: Fill missing early/preclinical phases, back up Phase 3 assets, balance maturity levels.
Key factors: dev stage gaps, search patterns, attrition data, disease continuity.
""",
    "Augmenting the Pipeline": """
Goal: Expand use of current molecules into new disease areas.

Focus: Indication expansion based on biological/pathway similarity, omics matching, clinical analogies.
Key factors: pathway overlap, clinical similarity, IP space, market need.
""",
    "Custom Query": """
Goal: Interpret the user’s custom query and suggest filter-level improvements for portfolio optimization.

Focus: Apply general pharma knowledge and make data-driven suggestions.
"""
}

# ------------------ STRATEGY ROUTER ------------------

def make_chain(strategy):
    prompt = ChatPromptTemplate.from_template(BASE_PROMPT_TEMPLATE)
    return (
        RunnableMap({
            "strategic_objective": lambda x: x["filters"]["strategic_objective"],
            "strategy_details": lambda x: STRATEGY_DETAILS.get(strategy, STRATEGY_DETAILS["Custom Query"]),
            "user_filters": lambda x: x["filters"],
            "competitor_data": lambda _: competitor_data
        })
        | prompt
        | llm
        | parser
    )

# ------------------ DATA LOAD ------------------

with open("competitor_data.json") as f:
    competitor_data = json.load(f)

user_input = {
    "filters": {
        "for_sale": "all",
        "asset_type": "molecule",
        "deal_value": {"min": 5, "max": 100},
        "asset_phenotype": {
            "stages_of_development": "Phase 1",
            "therapeutic_area": "Oncology",
            "indication": "Lung Cancer",
            "mechanism_of_action": "EGFR Inhibitor",
            "route_of_administration": "",
            "modality": "Small Molecule",
            "patent_expiry_yr": None
        },
        "company_details": {
            "company_type": "Biotech",
            "development_stage": "Clinical Stage",
            "headquarters": "USA",
            "financial_status": "Stable",
            "partner_status": "Unpartnered",
            "territory": ""
        },
        "peak_sales": {
            "1yr_sales_potential": 25,
            "5yr_sales_potential": None,
            "peak_sales": None
        },
        "strategic_objective": "how to increase my sales",  # Change this to test other strategies
        "custom_query_text": ""
    },
    "current_company_id": "your_company_id_123"
}

# ------------------ RUN STRATEGY AGENT ------------------

def route_to_chain(filters):
    strategy = filters["strategic_objective"]
    if strategy in STRATEGY_DETAILS:
        return make_chain(strategy)
    else:
        return make_chain("Custom Query")

if __name__ == "__main__":
    chain = route_to_chain(user_input["filters"])
    output = chain.invoke(user_input)

    print("\n=== STRATEGY RECOMMENDATIONS ===\n")
    for rec in output:
        print(f"Filter: {rec['filter']}")
        print(f"Suggested Change: {rec['suggested_change']}")
        print(f"Reason: {rec['reason']}\n")

    # Save to JSON
    output_file = "strategy_recommendations.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Recommendations saved to: {output_file}")
