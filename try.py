import os
import json
from dotenv import load_dotenv

# --- Core LangChain Imports ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LLM and Vector Store Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


# In a real application, you would load these from actual files.
# example data
pfizer_data = """
Company: Pfizer
Portfolio:
- Drug: Liptor (Atorvastatin), Indication: Hypercholesterolemia, Stage: Marketed, Patent Expiry: 2011 (Generic), Performance: Historically strong, now facing generic competition.
- Drug: Ibrance (Palbociclib), Indication: HR+ Breast Cancer, Stage: Marketed, Patent Expiry: 2027, Performance: Strong market leader, but facing new competitors.
- Drug: Paxlovid (Nirmatrelvir/Ritonavir), Indication: COVID-19, Stage: Marketed (EUA), Patent Expiry: 2040s, Performance: High initial uptake, demand now waning.
- Drug: PF-06463922, Indication: ALK+ Non-Small Cell Lung Cancer, Stage: Phase 2, Risk Score: Medium, Notes: Stuck in Phase 2 for 3 years due to recruitment challenges.
- Drug: Elrexfio (Elranatamab), Indication: Multiple Myeloma, Modality: Bispecific Antibody, Stage: Marketed, Notes: New entrant, strong potential.
"""

merck_data = """
Company: Merck
Portfolio:
- Drug: Keytruda (Pembrolizumab), Indication: Multiple Cancers (Melanoma, NSCLC, etc.), Stage: Marketed, Patent Expiry: 2028, Performance: Dominant market leader in immuno-oncology.
- Drug: Januvia (Sitagliptin), Indication: Type 2 Diabetes, Stage: Marketed, Patent Expiry: 2022 (Generic), Performance: Facing significant generic erosion.
- Drug: Gardasil 9 (HPV Vaccine), Indication: HPV Prevention, Stage: Marketed, Patent Expiry: 2028, Performance: Strong global leader.
- Drug: MK-1084, Indication: KRAS G12C+ Cancers, Stage: Phase 1, Modality: Small Molecule, Notes: Promising early data, addressing a competitive but high-need area.
"""

objectives_data = """
Strategic Objectives for Portfolio Analysis:

1. Pruning the Portfolio:
Goal: Identify assets for de-prioritization.
Focus: Low ROI; clinically inferior (vs. Standard of Care); high risk (toxicity, ADMET); market underperformance; short patent expiry; oversaturated markets.

2. Diversification:
Goal: Broaden the portfolio into new areas.
Focus: Gaps in disease space (new ICD/MeSH codes); new/orthogonal Mechanisms of Action (MoA); new modalities (peptides, biologics); unmet market needs or orphan drug opportunities.

3. Filling the Pipeline:
Goal: Strengthen the pipeline by filling gaps.
Focus: Gaps in development stages (e.g., no Phase 1); lack of backup candidates for late-stage assets; imbalance in asset maturity (short vs. long-term).

4. Augmenting the Pipeline / New Indications:
Goal: Suggest new indications for assets.
Focus: Biological pathway overlap; literature-mined links; similarities in clinical trial design; molecular signature matching between asset and new disease.
"""

def setup_vector_database():
    """
    Initializes the ChromaDB vector store, processing and embedding documents.
    This function should only be run once unless the source documents change.
    """
    print("Setting up the ChromaDB vector store...")
    db_path = "portfolio_db"
    
    # Create Document objects with metadata
    docs = [
        Document(page_content=pfizer_data, metadata={"doc_type": "portfolio", "company_name": "Pfizer"}),
        Document(page_content=merck_data, metadata={"doc_type": "portfolio", "company_name": "Merck"}),
        Document(page_content=objectives_data, metadata={"doc_type": "objectives"}),
    ]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Initialize embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_model, 
        persist_directory=db_path
    )
    
    print(f"Vector store created/loaded with {vectorstore._collection.count()} documents.")
    return vectorstore



# The prompt(needs most of the fixing)
SYSTEM_PROMPT = """
You are an expert pharmaceutical portfolio strategy analyst. Your goal is to generate a comprehensive, structured JSON report analyzing a given company's drug portfolio. You MUST use the provided retriever tool to fetch all necessary documents.

### Instructions:
1.  **Identify the Target Company**: From the user's input, identify the company to be analyzed.
2.  **Retrieve All Data**: You MUST use the `knowledge_base_retriever` tool to get three key pieces of information:
    - The target company's portfolio (e.g., search for 'portfolio for Pfizer').
    - The portfolios of all other companies to use as a benchmark (e.g., search for 'portfolio for Merck').
    - The strategic objectives document (e.g., search for 'strategic objectives').
3.  **Analyze and Report**: Based on the retrieved information, perform a detailed analysis for EACH of the four strategic objectives. For each objective, you MUST generate a list of recommendations with clear justifications.
4.  **Output Format**: Your FINAL output MUST be a single, well-structured JSON object. Do not include any text, explanations, or markdown formatting like ```json before or after the JSON block.

### Strategic Objectives & JSON Structure:

**1. Pruning the Portfolio (deprioritization_report)**
   - **Criteria**: Low ROI, clinically inferior, high-risk score, market underperformance, short patent expiry, oversaturated market.
   - **JSON Object Structure**: `{"asset_name": "string", "reason_for_deprioritization": "string", "recommended_action": "string"}`

**2. Diversification (diversification_opportunities)**
   - **Criteria**: Gaps in disease space, new/orthogonal MoA, new modalities, unmet market needs.
   - **JSON Object Structure**: `{"area_of_opportunity": "string", "suggested_modality": "string", "justification": "string"}`

**3. Filling the Pipeline (pipeline_gaps_report)**
   - **Criteria**: Gaps in development stages, lack of backup candidates, imbalance in asset maturity.
   - **JSON Object Structure**: `{"pipeline_gap": "string", "suggested_asset_profile": "string", "justification": "string"}`

**4. Augmenting the Pipeline / New Indications (new_indication_opportunities)**
   - **Criteria**: Biological pathway overlap, literature-mined links, clinical trial similarities.
   - **JSON Object Structure**: `{"asset_name": "string", "suggested_new_indication": "string", "evidence_and_justification": "string"}`

### Final JSON Structure Example:
```json
{
  "analyzed_company": "[Company Name]",
  "deprioritization_report": [ ... list of pruning objects ... ],
  "diversification_opportunities": [ ... list of diversification objects ... ],
  "pipeline_gaps_report": [ ... list of pipeline filling objects ... ],
  "new_indication_opportunities": [ ... list of augmentation objects ... ]
}
"""

def main():

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file and add it.")


    vector_store = setup_vector_database()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


    retriever_tool = create_retriever_tool(
        retriever,
        "knowledge_base_retriever",
        "Searches and retrieves information from the knowledge base about company drug portfolios and strategic objectives. Use this to get all the data you need for your analysis."
    )
    tools = [retriever_tool]


    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", 
        temperature=0.1,
        model_kwargs={"response_mime_type": "application/json"}
    )

    # agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), 
    ])


    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )

    # --- 3. Run the Agent ---
    print("\n--- Portfolio Analysis Agent is Ready ---")
    user_query = input("Enter your query (e.g., 'Analyze the portfolio for Pfizer'): ")

    if user_query:
        print("\nAnalyzing... please wait.\n")
        response = agent_executor.invoke({"input": user_query})
        
        
        try:
            
            json_output = json.loads(response['output'])
            print("\n--- Analysis Report ---")
            print(json.dumps(json_output, indent=2))
        except (json.JSONDecodeError, KeyError) as e:
            print("\n--- Agent Response (could not parse as JSON) ---")
            print(f"Error: {e}")
            print(response['output'])


if __name__ == "main":
    main()