"""
🚀 MAIN API - CUSTOMER CHURN PREDICTION
Orchestrates all API components (routes, models, config, utils)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import warnings

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, HOST, PORT, LOG_LEVEL
from app.routes import router

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 🎨 FASTAPI APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ═══════════════════════════════════════════════════════════════════════════
# 🔗 MIDDLEWARE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Add CORS middleware - Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# 📌 INCLUDE ROUTES
# ═══════════════════════════════════════════════════════════════════════════

app.include_router(router)

# ═══════════════════════════════════════════════════════════════════════════
# 🚀 RUN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting {API_TITLE}")
    print(f"{'='*70}")
    print(f"📍 Host: {HOST}")
    print(f"🔌 Port: {PORT}")
    print(f"📊 API Version: {API_VERSION}")
    print(f"\n📖 Documentation: http://{HOST}:{PORT}/docs")
    print(f"{'='*70}\n")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL
    )
