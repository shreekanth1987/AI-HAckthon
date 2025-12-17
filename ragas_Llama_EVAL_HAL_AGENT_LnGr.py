import os
import sys
import pandas as pd
from pathlib import Path
from app import graph
from datetime import datetime
# from langchain_ollama import OllamaLLM, OllamaEmbeddings  # Updated imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader,
)

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, AspectCritic,RubricsScore
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
import json
import httpx

from deepeval import evaluate as d_evaluate
from deepeval.metrics import HallucinationMetric , GEval
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval import Rubric

client = httpx.Client(verify=False)
# === LOGGING SETUP ===
class Logger:
    """Logger class to write to both console and log file."""
    def __init__(self, log_folder="./execution_logs"):
        # Create log folder if it doesn't exist
        os.makedirs(log_folder, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
        log_filename = f"dry_run_{timestamp}.log"
        self.log_path = os.path.join(log_folder, log_filename)
        
        # Open log file
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.terminal = sys.stdout
        
        self.write(f"Log file created: {self.log_path}\n")
        self.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.write("="*60 + "\n")
    
    def write(self, message):
        """Write to both terminal and log file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write
    
    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        self.write("\n" + "="*60 + "\n")
        self.write(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.close()

# Initialize logger
logger = Logger()
sys.stdout = logger
sys.stderr = logger

# --- CONFIGURATION ---
EVALUATOR_MODEL_NAME = "azure/genailab-maas-gpt-4o-mini" 
EMBEDDING_MODEL_NAME = "azure/genailab-maas-text-embedding-3-large"
ALL_MODELS_BASE_URL = "https://genailab.tcs.in"
RESOURCE_FOLDER = "./resource"
FAISS_INDEX_PATH = "./faiss_index"
# GROUND_TRUTH_FILE = "./gpt_hallucinate.json"
GROUND_TRUTH_FILE = "./rag_hallucinate.json"


# 1. Initialize Models with updated imports
print("Initializing models...")
# llm = OllamaLLM(
#     model=LLM_MODEL_NAME,
#     base_url=OLLAMA_BASE_URL,
#     timeout=300
# )

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-4o-mini",
    api_key="sk-y7jfDp8eNdcLs0PichvPsg",
    http_client=client
)

testing_llm = graph 
# ChatOpenAI(
#     base_url="https://genailab.tcs.in",
#     model="azure_ai/genailab-maas-DeepSeek-R1",
#     api_key="sk-y7jfDp8eNdcLs0PichvPsg",
#     http_client=client
# )

evaluator_llm_instance = ChatOpenAI(
            base_url="https://genailab.tcs.in",
            model="azure/genailab-maas-gpt-4o-mini",
            api_key="sk-y7jfDp8eNdcLs0PichvPsg",
            http_client=client
        )

# embeddings = OllamaEmbeddings(
#     model=EMBEDDING_MODEL_NAME,
#     base_url=OLLAMA_BASE_URL
# )

embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key='sk-y7jfDp8eNdcLs0PichvPsg', # Replace with your actual API key
    http_client=client
)

# FIASS loader assertion
tiktoken_cache_dir = "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

# 2. Load Documents from Local Folder (FIXED)
def load_documents_from_folder(folder_path):
    """Load various document types from a folder."""
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"⚠️  Folder {folder_path} doesn't exist. Creating it...")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"📁 Please add your documents to {folder_path} and run again.")
        logger.close()
        return documents
    
    print(f"\n📂 Loading documents from {folder_path}...")
    
    # Load TXT files
    txt_files = list(folder.glob("*.txt"))
    for file in txt_files:
        try:
            print(f"  Loading: {file.name}")
            # Read with UTF-8 encoding and handle errors
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if content.strip():  # Only add if not empty
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file.name, "type": "txt"}
                ))
        except Exception as e:
            print(f"  ⚠️  Error loading {file.name}: {e}")
    
    # Load PDF files
    pdf_files = list(folder.glob("*.pdf"))
    for file in pdf_files:
        try:
            print(f"  Loading: {file.name}")
            loader = PyPDFLoader(str(file))
            pdf_docs = loader.load()
            if pdf_docs:
                documents.extend(pdf_docs)
        except Exception as e:
            print(f"  ⚠️  Error loading {file.name}: {e}")
    
    print(f"✓ Loaded {len(documents)} document(s)")
    
    # Debug: Print sample content
    if documents:
        print(f"  Sample content from first doc: {documents[0].page_content[:200]}...")
    
    return documents

# 3. Create or Load FAISS Vector Store (FIXED)
def create_or_load_faiss_index(documents, index_path, embeddings, force_recreate=False):
    """Create new FAISS index or load existing one."""
    
    if os.path.exists(index_path) and not force_recreate:
        print(f"\n📦 Loading existing FAISS index from {index_path}...")
        try:
            vector_store = FAISS.load_local(
                index_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✓ Loaded FAISS index with {vector_store.index.ntotal} vectors")
            return vector_store
        except Exception as e:
            print(f"⚠️  Error loading index: {e}")
            print("Creating new index...")
    
    print(f"\n🔨 Creating new FAISS index...")
    
    if not documents:
        raise ValueError("No documents provided to create FAISS index!")
    
    # Chunk documents - FIXED to handle both list and individual docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split documents properly
    chunked_docs = []
    for doc in documents:
        # Ensure doc has content
        if hasattr(doc, 'page_content') and doc.page_content.strip():
            splits = text_splitter.split_documents([doc])
            chunked_docs.extend(splits)
    
    print(f"  Split into {len(chunked_docs)} chunks")
    
    if len(chunked_docs) == 0:
        raise ValueError("No chunks created! Documents might be empty or too short.")
    
    # Debug: Show sample chunks
    print(f"  Sample chunk: {chunked_docs[0].page_content[:100]}...")
    
    # Create FAISS index
    print(f"  Creating embeddings (this may take a while)...")
    try:
        vector_store = FAISS.from_documents(chunked_docs, embeddings)
        
        # Save index
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        print(f"✓ FAISS index saved to {index_path}")
        
        return vector_store
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")
        raise

# 4. Generate Ground Truth (Semi-Automated)
def generate_ground_truth_questions(llm, documents, num_questions=3):
    """Use LLM to generate questions and answers from documents."""
    
    print(f"\n🤖 Generating {num_questions} question-answer pairs...")
    
    # Sample content from documents
    sample_content = "\n\n".join([
        doc.page_content[:800] 
        for doc in documents[:3] 
        if hasattr(doc, 'page_content')
    ])
    
    prompt = f"""Based on the following content, generate {num_questions} diverse questions that can be answered using this information. 

Content:
{sample_content}

Generate questions in this EXACT JSON format (no extra text):
[
  {{"question": "What is...", "answer": "The answer is..."}},
  {{"question": "How does...", "answer": "It works by..."}}
]

Only output valid JSON, nothing else."""

    try:
        response = llm.invoke(prompt)
        print(f"  LLM response preview: {response[:200]}...")
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            qa_pairs = json.loads(json_match.group())
            print(f"✓ Generated {len(qa_pairs)} Q&A pairs")
            return qa_pairs
        else:
            print("⚠️  Could not parse LLM response as JSON")
            return []
    except Exception as e:
        print(f"⚠️  Error generating questions: {e}")
        return []

def save_ground_truth(qa_pairs, filepath):
    """Save ground truth to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"✓ Ground truth saved to {filepath}")

def load_ground_truth(filepath):
    """Load ground truth from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} Q&A pairs from {filepath}")
        return data
    return None

def create_manual_ground_truth_template():
    """Create template for manual ground truth creation."""
    template = [
        {
            "question": "Your question here?",
            "answer": "Expected answer here"
        },
        {
            "question": "Another question?",
            "answer": "Another answer"
        },
        {
            "question": "Third question?",
            "answer": "Third answer"
        }
    ]
    return template

# === MAIN EXECUTION ===

print("="*60)
print("RAG EVALUATION WITH LOCAL DOCUMENTS")
print("="*60)

# # Load documents
# documents = load_documents_from_folder(RESOURCE_FOLDER)

# if not documents:
#     print("\n" + "="*60)

#     # Close logger at the end
#     logger.close()
#     print("NO DOCUMENTS FOUND!")
#     print("="*60)
#     print(f"\nPlease add documents to: {RESOURCE_FOLDER}")
#     print("\nSupported formats:")
#     print("  - .txt (text files)")
#     print("  - .pdf (PDF documents)")
#     print("\nThen run this script again.")
#     logger.close()
#     exit()

# # Check if documents have content
# valid_docs = [doc for doc in documents if hasattr(doc, 'page_content') and doc.page_content.strip()]
# print(f"\n✓ {len(valid_docs)} valid documents with content")

# if len(valid_docs) == 0:
#     print("❌ All documents are empty! Please add documents with content.")
#     logger.close()
#     exit()

# # Create/Load FAISS index
# try:
#     vector_store = create_or_load_faiss_index(
#         valid_docs, 
#         FAISS_INDEX_PATH, 
#         embeddings,
#         force_recreate=False  # Set to True to rebuild
#     )
# except Exception as e:
#     print(f"\n❌ Failed to create FAISS index: {e}")
#     print("\nTroubleshooting:")
#     print("1. Check that documents have readable content")
#     print("2. Try setting force_recreate=True")
#     print("3. Check Ollama is running: ollama list")
#     logger.close()
#     exit()

# # Setup retriever
# retriever = vector_store.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,
#         "fetch_k": 6,
#         "lambda_mult": 0.7
#     }
# )

# # Build RAG Chain
# template = """Answer the question based ONLY on the following context. Be specific and comprehensive.

# Context:
# {context}

# Question: {query}

# Answer:"""

# prompt = ChatPromptTemplate.from_template(template)

# def format_docs(relevant_docs):
#     return "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

# rag_chain = (
#     {
#         "context": retriever | format_docs, 
#         "query": RunnablePassthrough()
#     }
#     | prompt
#     | testing_llm 
#     | StrOutputParser()
# )

# === GROUND TRUTH HANDLING ===

print("\n" + "="*60)
print("GROUND TRUTH SETUP")
print("="*60)

ground_truth = load_ground_truth(GROUND_TRUTH_FILE)

# === EVALUATION STARTS ===

if ground_truth:
    # print("\n" + "="*60)
    # print("RUNNING RAG EVALUATION")
    # print("="*60)
    
    # # Generate RAG outputs
    # ragas_data = []
    # print(f"\nGenerating answers for {len(ground_truth)} questions...")
    
    # for i, item in enumerate(ground_truth):
    #     query = item["question"]
    #     print(f"\n[{i+1}/{len(ground_truth)}] {query}")
        
    #     try:
    #         # Retrieve contexts
    #         retrieved_docs = testing_llm.invoke(query)
    #         contexts = [doc.page_content for doc in retrieved_docs]
            
    #         # Generate answer
    #         response = testing_llm.invoke(query)
    #         print(f"  Answer: {response[:100]}...")
            
    #         ragas_data.append({
    #             "question": query,
    #             "answer": response,
    #             "contexts": contexts,
    #             "reference": item["answer"]
    #         })
    #     except Exception as e:
    #         print(f"  ⚠️  Error processing question: {e}")
    #         continue
    
    # if not ragas_data:
    #     print("\n❌ No data to evaluate!")
    #     logger.close()
    #     exit()
    
    # # Evaluate
    # evaluation_dataset = Dataset.from_list(ragas_data)
    
    # print("\n" + "="*60)
    # print("EVALUATING WITH RAGAS...")
    # print("="*60)
    

    # evaluator_llm = LangchainLLMWrapper(evaluator_llm_instance)

    # # you can init the metric with the evaluator llm
    # hallucinations_binary = AspectCritic(
    #     name="hallucinations_binary",
    #     definition="Did the model hallucinate or add any information that was not present in the retrieved context?",
    #     llm=evaluator_llm,
    # )

    # # HAL rubric and evaluator LLM and evaluate the dataset.
    # hal_rubric = {
    #             "score1_description": "There is no hallucination in the response. All the information in the response is present in the retrieved context.",
    #             "score2_description": "There are no factual statements that are not present in the retrieved context but the response is not fully accurate and lacks important details.",
    #             "score3_description": "There are many factual statements that are not present in the retrieved context.",
    #             "score4_description": "The response contains some factual errors and lacks important details.",
    #             "score5_description": "The model adds new information and statements that contradict the retrieved context.",
    #         }
    # hallucinations_rubric = RubricsScore(
    #     name="hallucinations_rubric", llm=evaluator_llm, rubrics=hal_rubric
    # )
    

    # ##################################################################### 
    # # Custom Metrics for Finding out Hallucination as per Ragas approach
    # #####################################################################

    # # we are going to create a dataclass that subclasses `MetricWithLLM` and `SingleTurnMetric`
    # from dataclasses import dataclass, field

    # # import the base classes
    # from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
    # from ragas.metrics import Faithfulness

    # # import types
    # import typing as t
    # from ragas.callbacks import Callbacks
    # from ragas.dataset_schema import SingleTurnSample


    # @dataclass
    # class HallucinationsMetric(MetricWithLLM, SingleTurnMetric):
    #     # name of the metric
    #     name: str = "hallucinations_metric"
    #     # we need to define the required columns for the metric
    #     _required_columns: t.Dict[MetricType, t.Set[str]] = field(
    #         default_factory=lambda: {
    #             MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
    #         }
    #     )

    #     def __post_init__(self):
    #         # init the faithfulness metric
    #         self.faithfulness_metric = Faithfulness(llm=self.llm)

    #     async def _single_turn_ascore(
    #         self, sample: SingleTurnSample, callbacks: Callbacks
    #     ) -> float:
    #         faithfulness_score = await self.faithfulness_metric.single_turn_ascore(
    #             sample, callbacks
    #         )
    #         return 1 - faithfulness_score
        
    # # Custom Metrics for Finding out Hallucination ends here`
    
    # hallucinations_metric = HallucinationsMetric(llm=evaluator_llm)

    # result = evaluate(
    #         evaluation_dataset,
    #         metrics=[hallucinations_metric, hallucinations_rubric, hallucinations_binary] ,
    #         llm=evaluator_llm,
    #         embeddings=embeddings,
    #         raise_exceptions=False
    #     )
    
    # # result = evaluate(
    # #     dataset=evaluation_dataset,
    # #     metrics=[
    # #         faithfulness,
    # #         answer_relevancy,
    # #         context_recall,
    # #         context_precision
    # #     ],
    # #     llm=evaluator_llm, 
    # #     embeddings=embeddings,
    # #     # batch_size=1,
    # #     raise_exceptions=False
    # # )
    
    # # Display Results
    # print("\n" + "="*60)
    # print("RAGAS EVALUATION RESULTS")
    # print("="*60)
    
    # scores = result.to_pandas().mean(numeric_only=True)

    # print(scores)

    # print(result.to_pandas())
    
    # print("\n📊 Overall Scores:")
    # print(f"  Hallucination Metric:       {scores['hallucinations_metric']}")
    # print(f"  Hallucination Rubic:   {scores['hallucinations_rubric']}")
    # print(f"  Hallucination Binary:     {scores['hallucinations_binary']}")

    
    # # print("\n📋 Per-Question Results:")
    # result_df = result.to_pandas()
    # timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
    # results_folder = Path("./hallucination_results")
    # results_folder.mkdir(exist_ok=True)  # Create folder if doesn't exist
    # results_filename = results_folder / f"ragas_hallucination_results{timestamp}.csv"
    # result_df.to_csv(results_filename, index=False)
    # # for idx, row in result_df.iterrows():
    # #     print(f"\nQuestion {idx + 1}: {ground_truth[idx]['question'][:60]}...")
    # #     print(f"  Faithfulness:      {row['faithfulness']:.2%}")
    # #     print(f"  Answer Relevancy:  {row['answer_relevancy']:.2%}")
    # #     print(f"  Context Recall:    {row['context_recall']:.2%}")
    # #     print(f"  Context Precision: {row['context_precision']:.2%}")
    
    # # Save results with timestamp in execution_results folder
    # # timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
    

    # # results_filename = results_folder / f"hallucination_results{timestamp}.csv"
    # # result_df.to_csv(results_filename, index=False)
    # # print(f"\n✓ Detailed results saved to {results_filename}")
    
    # print("\n" + "="*60)

    # Display Results

    ##################################################################### 
    # DEEP EVAL EVALUATION RESULTS
    #####################################################################
    print("\n" + "="*60)
    print("DEEP EVAL EVALUATION RESULTS")
    print("="*60)

    # Replace this with the actual documents that you are passing as input to your LLM.
    context=["Differentiating Evaluation Scope Drawing on the provided analogy, if Large Language Model (LLM) evaluation is likened to examining the performance of an engine, what critical complexity does LLM agent evaluation add, and why are standard LLM evaluation methods insufficient for agents?"]

    # Replace this with the actual output from your LLM application
    actual_output="The analogy clarifies that while LLM evaluation assesses the engine's performance, agent evaluation assesses a car’s performance comprehensively, as well as under various driving conditions. This adds complexity because LLM agents are not just assessed for text generation; they operate in dynamic, interactive environments where they must reason, plan, execute tools, leverage memory, and possibly collaborate with other agents or humans. Standard LLM evaluation is insufficient because it focuses on deterministic and static behavior, whereas LLM agents are inherently probabilistic and behave dynamically, requiring evaluation that incorporates natural language processing (NLP), human-computer interaction (HCI), and software engineering perspectives"

    # test_case = LLMTestCase(
    #     input="what critical complexity does LLM agent evaluation add, why are standard LLM evaluation methods insufficient for agents",
    #     actual_output=actual_output,
    #     context=context
    # )

    ###### WRAPPER CLASS TO INVOKE CUSTOM LLM MODEL FOR EVALUATION #########
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    class AzureOpenAI(DeepEvalBaseLLM):
        def __init__(
            self,
            model
        ):
            self.model = model

        def load_model(self):
            return self.model

        def generate(self, prompt: str) -> str:
            chat_model = self.load_model()
            return chat_model.invoke(prompt).content

        async def a_generate(self, prompt: str) -> str:
            chat_model = self.load_model()
            res = await chat_model.ainvoke(prompt)
            return res.content

        def get_model_name(self):
            return "Custom Azure OpenAI Model"
    
    deep_evaluator_llm = AzureOpenAI(model=evaluator_llm_instance)
    # metric = HallucinationMetric(threshold=0.3, model=deep_evaluator_llm,include_reason=True)


    # === GENERATING TEST CASES AND RESULTS DATA FOR DEEPEVAL ===
    if ground_truth:
        print("\n" + "="*60)
        print("RUNNING RAG EVALUATION WITH DEEPEVAL")
        print("="*60)
        
        # Generate RAG outputs
        deepeval_test_cases = []
        deepeval_results_data = []
        print(f"\nGenerating answers for {len(ground_truth)} questions...")
    
        for i, item in enumerate(ground_truth):
            query = item["question"]
            print(f"\n[{i+1}/{len(ground_truth)}] {query}")
            
            try:
                # Retrieve contexts
                retrieved_docs = testing_llm.invoke(query)
                contexts = [doc.page_content for doc in retrieved_docs]
                
                # Generate answer
                response = testing_llm.invoke(query)
                print(f"  Answer: {response[:100]}...")
                
                # Create DeepEval test case
                test_case = LLMTestCase(
                    input=query,
                    actual_output=response,
                    expected_output=item["answer"],
                    retrieval_context=contexts, # Recommended RAG parameter
                    context=contexts             # The parameter the HallucinationMetric is currently requiring
                )
                deepeval_test_cases.append(test_case)
                
                deepeval_results_data.append({
                    "question": query,
                    "expected_answer": item["answer"],
                    "generated_answer": response,
                    "retrieved_docs": len(contexts),
                    "first_context": contexts[0][:100] if contexts else "N/A"
                })
                
                print(f"Printing deepeval_results_data:\n{deepeval_results_data}")
            except Exception as e:
                print(f"Error processing question: {e}")
                continue
        
            # Create metrics for DeepEval
            print("\n" + "="*60)
            print("INITIALIZING DEEPEVAL METRICS")
            print("="*60)

    
        # === DEEPEVAL METRIC CONFIGURATION FOR MODEL EVALUATION ===
        metrics = {
                    'hallucination': HallucinationMetric(
                        threshold=0.5,
                        model=deep_evaluator_llm,
                        include_reason=True
                    ),
                    'correctness': GEval(
                        name="Correctness",
                        model=deep_evaluator_llm,
                        evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                        evaluation_steps=[
                            "Determine whether the actual output is factually correct based on the expected output."
                        ],
                        threshold=0.5,
                        # include_reason=True
                        rubric=[
                            Rubric(score_range=(0,2), expected_outcome="Factually incorrect."),
                            Rubric(score_range=(3,6), expected_outcome="Mostly correct."),
                            Rubric(score_range=(7,9), expected_outcome="Correct but missing minor details."),
                            Rubric(score_range=(10,10), expected_outcome="Factually Correct"),
                            ]
                        )
                }
        
        print("Deepeval Metrics configured:")
        print(metrics)
        # for metric_name,metric_value in enumerate(metrics):
        #     print(f"  ✓ {metric_name} : {metric_value}")

        # Run evaluation
        print("\n" + "="*60)
        print("EVALUATING WITH DEEPEVAL METRICS")
        print("="*60)
        
        try:
            results = d_evaluate(
                test_cases=deepeval_test_cases,
                metrics=[
                    metrics['hallucination'],
                    metrics['correctness']
                ]
            )
            
            print("\nEvaluation completed successfully!")
            print(results)
            
        except Exception as e:
            print(f"\nEvaluation completed with issues: {e}")
            print("Continuing to display available results...")
        
        # Display Results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
    

        # Replace the DeepEval evaluation section (around line 520-620) with this:

        # Run evaluation
        print("\n" + "="*60)
        print("EVALUATING WITH DEEPEVAL METRICS")
        print("="*60)

        # Store results separately (can't add to Pydantic LLMTestCase)
        test_case_results = []

        print("\nEvaluating test cases individually...")

        for i, test_case in enumerate(deepeval_test_cases):
            print(f"\nProcessing test case {i + 1}/{len(deepeval_test_cases)}...")
            
            try:
                # Create fresh metrics for each test case to avoid state issues
                hallucination_metric = HallucinationMetric(
                    threshold=0.5,
                    model=deep_evaluator_llm,
                    include_reason=True,
                    async_mode=False  # Disable async to avoid ainvoke issues
                )
                
                correctness_metric = GEval(
                    name="Correctness",
                    model=deep_evaluator_llm,
                    evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                    evaluation_steps=[
                        "Determine whether the actual output is factually correct based on the expected output."
                    ],
                    threshold=0.5,
                    async_mode=False,  # Disable async to avoid ainvoke issues
                    rubric=[
                        Rubric(score_range=(0,2), expected_outcome="Factually incorrect."),
                        Rubric(score_range=(3,6), expected_outcome="Mostly correct."),
                        Rubric(score_range=(7,9), expected_outcome="Correct but missing minor details."),
                        Rubric(score_range=(10,10), expected_outcome="Factually Correct"),
                    ]
                )
                
                # Measure hallucination
                print(f"  Measuring hallucination...")
                hallucination_metric.measure(test_case)
                print(f"    Score: {hallucination_metric.score}")
                
                # Measure correctness
                print(f"  Measuring correctness...")
                correctness_metric.measure(test_case)
                print(f"    Score: {correctness_metric.score}")
                
                # Store results in separate dictionary
                test_case_results.append({
                    'test_case': test_case,
                    'metrics': {
                        'hallucination': {
                            'score': hallucination_metric.score,
                            'reason': getattr(hallucination_metric, 'reason', None),
                            'threshold': hallucination_metric.threshold,
                            'success': hallucination_metric.is_successful()
                        },
                        'correctness': {
                            'score': correctness_metric.score,
                            'reason': getattr(correctness_metric, 'reason', None),
                            'threshold': correctness_metric.threshold,
                            'success': correctness_metric.is_successful()
                        }
                    }
                })
                
            except Exception as e:
                print(f"  Error evaluating test case {i + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\nEvaluation completed!")

        # Display Results
        print("\n" + "="*60)
        print("HALLUCINATION EVALUATION RESULTS FROM DEEP EVAL")
        print("="*60)

        # Calculate and display metrics
        hal_results_summary = []

        for i, result in enumerate(test_case_results):
            test_case = result['test_case']
            metrics = result['metrics']
            
            print(f"\n{'='*60}")
            print(f"Question {i + 1}: {test_case.input[:80]}...")
            print(f"{'='*60}")
            print(f"Expected: {test_case.expected_output[:100]}...")
            print(f"Generated: {test_case.actual_output[:100]}...")
            print(f"Retrieved {len(test_case.retrieval_context)} context(s)")
            
            result_entry = {
                "question_num": i + 1,
                "question": test_case.input,
                "expected_output": test_case.expected_output,
                "actual_output": test_case.actual_output,
                "num_contexts": len(test_case.retrieval_context)
            }
            
            # Display metrics
            print(f"\n  📊 Metrics:")
            for metric_name, metric_data in metrics.items():
                score = metric_data['score']
                reason = metric_data['reason']
                success = metric_data['success']
                
                result_entry[f'{metric_name}_score'] = score
                result_entry[f'{metric_name}_success'] = success
                
                # Display score
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {metric_name.upper()}: {score:.4f} {status}")
                
                # Display reason if available
                if reason:
                    print(f"    Reason: {reason[:200]}...")
            
            hal_results_summary.append(result_entry)

        # Calculate overall statistics
        if hal_results_summary:
            print("\n" + "="*60)
            print("OVERALL STATISTICS")
            print("="*60)
            
            # Extract metric scores
            hallucination_scores = [
                entry.get('HallucinationMetric') 
                for entry in hal_results_summary 
                if entry.get('HallucinationMetric') is not None
            ]
            correctness_scores = [
                entry.get('Correctness') 
                for entry in hal_results_summary 
                if entry.get('Correctness') is not None
            ]
            
            if hallucination_scores:
                avg_hallucination = sum(hallucination_scores) / len(hallucination_scores)
                print(f"Average Hallucination Score: {avg_hallucination:.4f}")
                print(f"  (Lower is better - 0 means no hallucination)")
            
            if correctness_scores:
                avg_correctness = sum(correctness_scores) / len(correctness_scores)
                print(f"Average Correctness Score: {avg_correctness:.4f}")
                print(f"  (Higher is better - 1.0 is perfect)")

        # Save results
        timestamp = datetime.now().strftime("%d_%m_%y_%H_%M")
        results_folder = Path("./hallucination_results")
        results_folder.mkdir(exist_ok=True)

        # Save summary as CSV
        results_filename = results_folder / f"deepeval_hallucination_results_{timestamp}.csv"
        results_df = pd.DataFrame(hal_results_summary)
        results_df.to_csv(results_filename, index=False)
        print(f"\n✅ Results saved to {results_filename}")

        # Also save detailed results with reasons
        detailed_filename = results_folder / f"deepeval_detailed_results_{timestamp}.json"
        detailed_results = []
        for result in test_case_results:
            test_case = result['test_case']
            metrics = result['metrics']
            
            case_details = {
                "question": test_case.input,
                "expected": test_case.expected_output,
                "actual": test_case.actual_output,
                "contexts": test_case.retrieval_context,
                "metrics": {
                    metric_name: {
                        "score": metric_data['score'],
                        "success": metric_data['success'],
                        "threshold": metric_data['threshold'],
                        "reason": metric_data['reason']
                    }
                    for metric_name, metric_data in metrics.items()
                }
            }
            detailed_results.append(case_details)

        with open(detailed_filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Detailed results with reasons saved to {detailed_filename}")