from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any

# Import the RAG function from your existing code
from main import run_cybersecurity_rag

app = FastAPI(title="Cybersecurity RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    metadata: Optional[Dict[str, Any]] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Run the RAG pipeline
        result = run_cybersecurity_rag(request.question)
        
        # Extract the generated response
        response = result.get("generation", "No response generated")
        
        # Create metadata with additional information
        metadata = {
            "documents_used": len(result.get("documents", [])),
            "web_search_performed": result.get("web_search") == "Yes",
            "final_question": result.get("question")  # This might be the reformulated question
        }
        
        return ChatResponse(
            answer=response,
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 