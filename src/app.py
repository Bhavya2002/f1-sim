"""
FastAPI Web Application: Formula 1 Race Strategy What-If Simulator.

Exposes:
- GET /: Serves main interactive dashboard HTML
- GET /api/race-info: Metadata, circuit info, baseline top 10 strategies
- POST /api/simulate: Lap-by-lap simulation endpoint with composable controls
- GET /api/validation/replay: Data handling replay check
- GET /api/validation/backtest: Model pace & deg backtest on held-out laps
- GET /api/validation/sanity: Sanity checks & data coverage checklist
"""

import os
import json
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.engine.scenario_manager import ScenarioManager
from src.engine.simulation_engine import SimulationEngine
from src.ai.assistant import StrategyAIAssistant
from src.validation.model_backtest import ModelBacktest
from src.validation.replay_check import ReplayCheck
from src.validation.sanity_checks import SanityChecks

app = FastAPI(title="Formula 1 Race Strategy What-If Simulator", version="1.0.0")

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize engine, assistant & validation singletons
engine = SimulationEngine()
scenario_mgr = ScenarioManager(engine)
ai_assistant = StrategyAIAssistant(engine)
replay_checker = ReplayCheck()
backtest_checker = ModelBacktest()
sanity_checker = SanityChecks(engine)


class StintItem(BaseModel):
    stint_number: int
    lap_start: int
    lap_end: int
    compound: str
    start_tyre_age: int = 0


class SimulationRequest(BaseModel):
    modified_driver: Optional[int] = None
    custom_stints: Optional[List[StintItem]] = None
    remove_safety_car: bool = False
    remove_vsc: bool = False
    insert_safety_car_lap: Optional[int] = None
    insert_safety_car_duration: int = 3
    insert_vsc_lap: Optional[int] = None
    insert_vsc_duration: int = 2
    replace_slow_stop: bool = False
    enable_traffic_model: bool = False
    use_nonlinear_tyre: bool = False


class ChatRequest(BaseModel):
    prompt: str
    current_state: Optional[dict] = None


class NLPPredictRequest(BaseModel):
    prompt: str


# Page Routes (Multi-Page Architecture)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "page": "studio",
        "race_info": engine.race_info
    })


@app.get("/ml-lab", response_class=HTMLResponse)
async def ml_lab(request: Request):
    return templates.TemplateResponse(request, "ml_lab.html", {
        "page": "ml_lab",
        "race_info": engine.race_info
    })


@app.get("/validation", response_class=HTMLResponse)
async def validation_page(request: Request):
    return templates.TemplateResponse(request, "validation.html", {
        "page": "validation",
        "race_info": engine.race_info
    })


@app.get("/api/race-info")
async def get_race_info():
    return {
        "race_info": engine.race_info,
        "baseline_stints": engine.baseline_stints,
        "official_classification": engine.official_top10,
    }


@app.post("/api/simulate")
async def simulate(req: SimulationRequest):
    custom_stints_dict = None
    if req.custom_stints:
        custom_stints_dict = [s.model_dump() for s in req.custom_stints]

    result = engine.run_simulation(
        modified_driver=req.modified_driver,
        custom_stints=custom_stints_dict,
        remove_safety_car=req.remove_safety_car,
        remove_vsc=req.remove_vsc,
        insert_safety_car_lap=req.insert_safety_car_lap,
        insert_safety_car_duration=req.insert_safety_car_duration,
        insert_vsc_lap=req.insert_vsc_lap,
        insert_vsc_duration=req.insert_vsc_duration,
        replace_slow_stop=req.replace_slow_stop,
        enable_traffic_model=req.enable_traffic_model,
        use_nonlinear_tyre=req.use_nonlinear_tyre,
    )
    return JSONResponse(content=result)


@app.post("/api/chat")
async def chat_assistant(req: ChatRequest):
    res = ai_assistant.process_query(user_text=req.prompt, current_state=req.current_state)
    return JSONResponse(content=res)


# Machine Learning Endpoints
@app.get("/api/ml/metrics")
async def get_ml_metrics():
    summary_file = "data/models/training_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            data = json.load(f)
            return JSONResponse(content=data)
    return JSONResponse(content={"status": "NOT_TRAINED", "models": {}})


@app.get("/api/ml/tyre-curves")
async def get_tyre_curves():
    return JSONResponse(content=engine.pace_model.get_tyre_degradation_curves())


@app.post("/api/ml/predict-nlp")
async def predict_nlp(req: NLPPredictRequest):
    if ai_assistant.nlp_model:
        return JSONResponse(content=ai_assistant.nlp_model.predict_intent(req.prompt))
    return JSONResponse(content={
        "intent": "UNKNOWN",
        "confidence": 0.0,
        "all_probabilities": {},
        "slots": {}
    })


@app.post("/api/ml/retrain")
async def retrain_all_models_api():
    from src.training.train_all import train_all_models
    summary = train_all_models()
    engine.pace_model._load_nonlinear_model()
    engine._load_traffic_model()
    ai_assistant._load_nlp_model()
    return JSONResponse(content={"status": "SUCCESS", "summary": summary})


@app.get("/api/validation/replay")
async def get_replay_validation():
    return JSONResponse(content=replay_checker.run())


@app.get("/api/validation/backtest")
async def get_backtest_validation():
    return JSONResponse(content=backtest_checker.run())


@app.get("/api/validation/sanity")
async def get_sanity_validation():
    return JSONResponse(content=sanity_checker.run())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
