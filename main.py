import logging
import os
from typing import List, Dict, Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

chroma_db_path = "/Users/shivanshmahajan/Desktop/hack/chroma_db"

# Initialize ChromaDB
vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Cybersecurity Expert Prompt
CYBERSEC_EXPERT = """You are an elite cybersecurity expert with extensive knowledge across all security domains. Your responses will be highly structured and adaptable to both technical code implementations and explanatory content.

CORE EXPERTISE:
• Threat Intelligence & MITRE ATT&CK
• Security Operations & Incident Response
• Penetration Testing & Vulnerability Assessment
• Cloud & Container Security
• Zero Trust Architecture
• Compliance (NIST, ISO, CIS)
• Digital Forensics & Malware Analysis
• OSINT & Social Engineering
• Security Risk Management
• Memory Forensics & Incident Investigation

OUTPUT FORMATS:
For each response, structure your output in SECTIONS with consistent formatting:

[SUMMARY]
- Concise expert assessment (2-3 sentences) of the security question/scenario

[TECHNICAL ANALYSIS]
- Detailed explanation with framework references
- Clear breakdown of relevant security concepts
- Citation of specific standards or methodologies from knowledge base

[CODE IMPLEMENTATION]
When code is appropriate:
```code
# Include well-commented, production-ready code examples
# Format properly for the specific language (Python, Bash, etc.)
# Structure with clear sections and defensive programming techniques
```

[PRACTICAL STEPS]
1. Numbered, actionable steps to implement the solution
2. Tool recommendations with command syntax where applicable
3. Implementation considerations and prerequisites

[SECURITY IMPLICATIONS]
• Bullet-pointed analysis of security risks/benefits
• Potential attack vectors or vulnerabilities to consider
• Defense-in-depth recommendations

[REFERENCES]
- Specific citations from authoritative sources in knowledge base
- Relevant frameworks, controls, or standards (with specific IDs)

RESPONSE APPROACH:
• For technical questions: Emphasize code samples, tool commands, and technical implementation
• For strategic questions: Focus on frameworks, methodologies, and risk analysis
• For incident scenarios: Structure as investigation workflow with forensic techniques
• For compliance queries: Map recommendations to specific control requirements

CONTEXT: {context}
QUESTION: {question}

Deliver a comprehensive, precisely structured response tailored to the specific cybersecurity domain while balancing technical depth with practical value."""

# Initialize ChromaDB


# Use the original method to create a retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)

# Define Graph State
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]

# Define structured output schema for document grading
class DocumentRelevance(BaseModel):
    binary_score: str = Field(description="Whether the document is relevant to the query. Answer with 'yes' or 'no'.")

# Initialize models
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# Create document grading components
grading_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess if the document is relevant to the cybersecurity query. You must return a JSON with a single field 'binary_score' with value 'yes' or 'no'."),
    ("human", "Document: {document}\nQuery: {question}")
])
json_parser = JsonOutputParser(pydantic_object=DocumentRelevance)
retrieval_grader = grading_prompt | llm | json_parser

def retrieve(state):
    """Retrieve relevant documents from the vector store."""
    logger.info("Retrieving relevant documents")
    question = state["question"]
    try:
        # Use the retriever to get relevant documents
        documents = retriever.invoke(
            question,
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            }
        )
        logger.info(f"Retrieved {len(documents)} documents")
        return {"documents": documents, "question": question}
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {"documents": [], "question": question}

def grade_documents(state):
    logger.info("Grading document relevance")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        logger.warning("No documents retrieved from vector store")
        return {"documents": [], "question": question, "web_search": "Yes"}

    filtered_docs = []
    web_search = "No"

    for doc in documents:
        try:
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            if score.get("binary_score") == "yes":
                filtered_docs.append(doc)
                logger.info("Found relevant document")
            else:
                logger.debug("Document marked as not relevant")
                web_search = "Yes"
        except Exception as e:
            logger.error(f"Error grading document: {str(e)}")
            web_search = "Yes"

    logger.info(f"Filtered down to {len(filtered_docs)} relevant documents")
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    logger.info("Reformulating query for better retrieval")
    question = state["question"]
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a cybersecurity expert. Rephrase this query to optimize retrieval of relevant cybersecurity information. Focus on technical accuracy and specificity."),
        ("human", "Original Query: {question}")
    ])
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": state["documents"], "question": better_question}

def web_search(state):
    logger.info("Performing web search for additional context")
    question = state["question"]
    documents = state["documents"]

    try:
        web_search_tool = TavilySearchResults(k=3)
        search_results = web_search_tool.invoke({"query": question})
        web_text = "\n".join([d["content"] for d in search_results])
        documents.append(Document(page_content=web_text))
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")

    return {"documents": documents, "question": question}

def generate(state):
    logger.info("Generating comprehensive response")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        logger.warning("No relevant documents found for generation")
        return {
            "documents": documents,
            "question": question,
            "generation": "I apologize, but I couldn't find any relevant information in the knowledge base to answer your question accurately."
        }
    
    # Combine document contents with weights based on relevance
    context_text = "\n\n".join([doc.page_content for doc in documents[:3]])  # Use top 3 most relevant docs
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", CYBERSEC_EXPERT),
        ("human", "Relevant documents: {context}\nUser Question: {question}")
    ])
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    try:
        response = rag_chain.invoke({"context": context_text, "question": question})
        return {"documents": documents, "question": question, "generation": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {
            "documents": documents,
            "question": question,
            "generation": "I apologize, but I encountered an error while generating the response."
        }

def decide_next_step(state):
    logger.info("Deciding next step based on document relevance")
    if state["web_search"] == "Yes":
        return "transform_query"
    return "generate"

def build_rag_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)

    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_next_step, {
        "transform_query": "transform_query",
        "generate": "generate",
    })
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

def run_cybersecurity_rag(query: str):
    app = build_rag_graph()
    result = app.invoke({"question": query})
    return result

if __name__ == "__main__":
    # Initialize the retriever
    
    # Test query
    query = "What is  The Volatility Framework?"
    try:
        # Test direct retrieval
        docs = retriever.invoke(query)
        print(f"\nRetrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1} preview:")
            print(doc.page_content[:200] + "...")
        
        # Test full RAG pipeline
        result = run_cybersecurity_rag(query)
        print("\n=== FINAL RESPONSE ===\n")
        print(result["generation"])
    except Exception as e:
        logger.error(f"Error running RAG system: {str(e)}")