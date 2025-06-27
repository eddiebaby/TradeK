"""FastAPI application for Technical Analysis system."""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .indicators import (
    ATRCalculator,
    BollingerBandsCalculator,
    EMACalculator,
    IndicatorCalculator,
    MACDCalculator,
    RSICalculator,
    SMApCalculator,
)
from .models import OHLCV


# Pydantic models for API requests/responses
class OHLCVRequest(BaseModel):
    """Request model for OHLCV data."""
    
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class IndicatorRequest(BaseModel):
    """Request model for indicator calculation."""
    
    ohlcv_data: List[OHLCVRequest]
    indicators: List[str] = Field(
        default=["RSI_14", "SMA_20", "EMA_10", "MACD_12_26_9"],
        description="List of indicators to calculate"
    )


class IndicatorResponse(BaseModel):
    """Response model for indicator calculation."""
    
    symbol: str
    timestamp: datetime
    indicator: str
    value: float
    components: Optional[Dict[str, float]] = None
    parameters: Dict[str, float]


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    
    status: str
    version: str
    available_indicators: List[str]
    uptime_seconds: float


# FastAPI application
app = FastAPI(
    title="Technical Analysis System",
    description="Production-grade technical analysis and stock analysis API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global indicator calculator
calculator = IndicatorCalculator()

# Initialize default indicators
def get_calculator() -> IndicatorCalculator:
    """Get the global indicator calculator."""
    if not calculator.indicators:
        # Register default indicators
        calculator.register("RSI_14", RSICalculator(period=14))
        calculator.register("RSI_21", RSICalculator(period=21))
        calculator.register("SMA_20", SMApCalculator(period=20))
        calculator.register("SMA_50", SMApCalculator(period=50))
        calculator.register("SMA_200", SMApCalculator(period=200))
        calculator.register("EMA_10", EMACalculator(period=10))
        calculator.register("EMA_21", EMACalculator(period=21))
        calculator.register("EMA_50", EMACalculator(period=50))
        calculator.register("MACD_12_26_9", MACDCalculator(fast=12, slow=26, signal=9))
        calculator.register("BB_20_2", BollingerBandsCalculator(period=20, std_dev=2))
        calculator.register("ATR_14", ATRCalculator(period=14))
    
    return calculator


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "Technical Analysis System API", "status": "operational"}


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/status", response_model=SystemStatusResponse)
async def system_status(calc: IndicatorCalculator = Depends(get_calculator)):
    """Get system status and available indicators."""
    return SystemStatusResponse(
        status="operational",
        version="0.1.0",
        available_indicators=list(calc.indicators.keys()),
        uptime_seconds=0.0,  # TODO: Implement actual uptime tracking
    )


@app.post("/indicators/calculate", response_model=List[IndicatorResponse])
async def calculate_indicators(
    request: IndicatorRequest,
    calc: IndicatorCalculator = Depends(get_calculator)
):
    """Calculate technical indicators for provided OHLCV data."""
    if not request.ohlcv_data:
        raise HTTPException(status_code=400, detail="OHLCV data is required")
    
    # Validate requested indicators
    invalid_indicators = [
        ind for ind in request.indicators 
        if ind not in calc.indicators
    ]
    if invalid_indicators:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid indicators: {invalid_indicators}. "
                   f"Available: {list(calc.indicators.keys())}"
        )
    
    results = []
    
    # Reset calculators for fresh calculation
    calc.reset_all()
    
    # Process each OHLCV data point
    for ohlcv_req in request.ohlcv_data:
        try:
            # Convert to domain model
            ohlcv = OHLCV(
                symbol=ohlcv_req.symbol,
                timestamp=ohlcv_req.timestamp,
                open=ohlcv_req.open,
                high=ohlcv_req.high,
                low=ohlcv_req.low,
                close=ohlcv_req.close,
                volume=ohlcv_req.volume,
            )
            
            # Calculate all indicators
            indicator_results = calc.calculate_all(ohlcv)
            
            # Filter to requested indicators and convert to response format
            for indicator_name in request.indicators:
                if indicator_name in indicator_results:
                    indicator_value = indicator_results[indicator_name]
                    
                    response = IndicatorResponse(
                        symbol=indicator_value.symbol,
                        timestamp=indicator_value.timestamp,
                        indicator=indicator_value.indicator,
                        value=float(indicator_value.value),
                        components={
                            k: float(v) for k, v in indicator_value.components.items()
                        } if indicator_value.components else None,
                        parameters={
                            k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in indicator_value.parameters.items()
                        },
                    )
                    results.append(response)
        
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing OHLCV data: {str(e)}"
            )
    
    return results


@app.get("/indicators/available", response_model=List[str])
async def get_available_indicators(calc: IndicatorCalculator = Depends(get_calculator)):
    """Get list of available technical indicators."""
    return list(calc.indicators.keys())


@app.post("/indicators/reset")
async def reset_indicators(calc: IndicatorCalculator = Depends(get_calculator)):
    """Reset all indicator calculators."""
    calc.reset_all()
    return {"message": "All indicators reset successfully"}


@app.get("/indicators/{indicator_name}/info")
async def get_indicator_info(
    indicator_name: str,
    calc: IndicatorCalculator = Depends(get_calculator)
):
    """Get information about a specific indicator."""
    indicator = calc.get_indicator(indicator_name)
    
    if not indicator:
        raise HTTPException(
            status_code=404,
            detail=f"Indicator '{indicator_name}' not found"
        )
    
    return {
        "name": indicator.name,
        "type": type(indicator).__name__,
        "description": indicator.__doc__ or "No description available",
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)