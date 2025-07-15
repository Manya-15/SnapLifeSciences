from flask import Flask, request, jsonify
import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

# --- Gemini API Key and Model ---
os.environ["GOOGLE_API_KEY"] = "key"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
parser = JsonOutputParser()

# ------------------ PROMPT TEMPLATE ------------------

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

def make_chain(strategy, competitor_data):
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

def route_to_chain(filters, competitor_data):
    strategy = filters["strategic_objective"]
    return make_chain(strategy, competitor_data)

# ------------------ API ENDPOINT ------------------

@app.route("/generate-strategy", methods=["POST"])
def generate_strategy():
    try:
        data = request.json
        with open("competitor_data.json") as f:
            competitor_data = json.load(f)
        

        chain = route_to_chain(data["filters"], competitor_data)
        output = chain.invoke(data)

        return jsonify({"recommendations": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ FLASK RUN ------------------

if __name__ == "__main__":
    app.run(debug=True)
