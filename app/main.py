# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Added
from app.api.agent import router as agent_router

app = FastAPI(title="LangGraph Agent Server - Python Edition")

# --- CORS Configuration ---
# Adjust origins for production if needed.
# The Next.js frontend (apps/web) typically runs on http://localhost:3000 during development.
# The LangGraph TypeScript server was on port 54367.
# The FastAPI server might run on port 8000 by default with uvicorn.
origins = [
    "http://localhost:3000", # Typical Next.js dev port
    "http://localhost:54367",# Old LangGraph TS server port (if frontend still points there sometimes)
    "*" # Allow all for broad development, tighten for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Allow cookies if your auth mechanism uses them
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)
# --- End CORS Configuration ---

@app.get("/")
async def root():
    return {"message": "LangGraph Agent Server is running"}

app.include_router(agent_router, prefix="/api/agent")
