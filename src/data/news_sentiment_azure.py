import time
import random
import numpy as np
from typing import Dict, List, Optional
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger()

class AzureOpenAIEngine:
    """
    Simulates interaction with Azure OpenAI (GPT-4o) for Financial Sentiment Analysis.
    In a real deployment, this would use `openai` library with Azure endpoints.
    Here, it simulates the *reasoning* logic based on data deviation.
    """
    
    def __init__(self):
        self.config = ConfigLoader.get("azure")
        self.model = self.config.get("openai_model", "gpt-4o")
        self.use_simulation = self.config.get("use_simulation", True)
        logger.info(f"AzureOpenAIEngine initialized (Model: {self.model}, Simulation: {self.use_simulation})")

    def _construct_prompt(self, event: str, actual: str, forecast: str, previous: str) -> str:
        return f"""
        Role: Senior Gold Trader (XAUUSD)
        Task: Analyze economic news impact.
        Event: {event}
        Actual: {actual}
        Forecast: {forecast}
        Previous: {previous}
        
        Output JSON: {{ "sentiment": float (-1.0 to 1.0), "reasoning": "string" }}
        """

    def analyze_event(self, event_name: str, actual: float, forecast: float, previous: Optional[float] = None) -> Dict:
        """
        Returns AI-driven sentiment analysis.
        Range: -1.0 (Bearish Gold) to 1.0 (Bullish Gold).
        Note: Positive Economic Data for USD usually means Bearish Gold (-1.0).
        """
        if not self.use_simulation:
            try:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key=self.config.get("api_key"),
                    api_version=self.config.get("api_version", "2024-02-15-preview"),
                    azure_endpoint=self.config.get("azure_endpoint")
                )
                
                prompt = self._construct_prompt(event_name, str(actual), str(forecast), str(previous) or "-")
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst JSON bot."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                
                # Parse logic would go here (simplified)
                content = response.choices[0].message.content
                import json
                try:
                    # Clean markdown code blocks if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    data = json.loads(content)
                    return {
                        "sentiment_score": float(data.get("sentiment", 0.0)),
                        "reasoning": data.get("reasoning", "No reasoning provided."),
                        "provider": "AzureOpenAI (Live)"
                    }
                except Exception as e:
                    logger.error(f"Failed to parse Azure response: {content} | Error: {e}")
                    # Fallback to simulation
                    pass
            except Exception as e:
                logger.error(f"Azure OpenAI Call Failed: {e}. Falling back to simulation.")
                # Fallback to simulation

        # --- SIMULATION LOGIC ---
        # Mimic GPT-4o reasoning
        
        event_lower = event_name.lower()
        deviation = actual - forecast
        
        # 1. Logic Rule Base (Prompt Engineering Mimic)
        is_usd_positive_feature = True # Default: Good data = Strong USD
        
        if "unemployment" in event_lower or "claims" in event_lower:
            is_usd_positive_feature = False # Good data (High Unemp) = Weak USD
            
        # 2. Magnitude Calculation
        # Calculate 'Standardized Shock' (Z-score proxy)
        # Avoid div by zero
        denom = max(abs(forecast), 0.1)
        shock_pct = deviation / denom
        
        # 3. Sentiment Mapping
        # If USD Strong -> Gold Bearish (Negative Score)
        sentiment_score = 0.0
        
        if is_usd_positive_feature:
            # Actual > Forecast => USD Up => Gold Down (-)
            if deviation > 0:
                sentiment_score = -0.8 # Bearish Shock
                reason = f"Stronger {event_name} ({actual} > {forecast}) boosts USD, pressuring Gold."
            elif deviation < 0:
                sentiment_score = 0.8 # Bullish Surprise
                reason = f"Weaker {event_name} ({actual} < {forecast}) weakens USD, lifting Gold."
        else:
            # (Unemployment) Actual > Forecast => USD Down => Gold Up (+)
            if deviation > 0:
                sentiment_score = 0.7 
                reason = f"Higher {event_name} points to economic cooling; Dollar weakens, Gold rallies."
            elif deviation < 0:
                sentiment_score = -0.7
                reason = f"Lower {event_name} suggests tight labor market; Fed hawkish, Gold falls."
        
        # 4. Add "AI Variability" (Nuance)
        # Real AI isn't binary. It considers context.
        # We add slight random noise to simulate 'uncertainty' or 'nuance'
        noise = random.uniform(-0.05, 0.05)
        sentiment_score = np.clip(sentiment_score + noise, -1.0, 1.0)
        
        # 5. Decay/Nuance based on shock magnitude
        if abs(shock_pct) < 0.1: # Small deviation
            sentiment_score *= 0.5 # Muted reaction
            reason = "Data inline with expectations. Market impact neutral/muted."
            
        # Mock Latency
        # time.sleep(0.1) 
        
        logger.debug(f"[AzureAI] {event_name}: Score {sentiment_score:.2f} | {reason}")
        
        return {
            "sentiment_score": round(sentiment_score, 2),
            "reasoning": reason,
            "provider": "AzureOpenAI (Sim)"
        }

if __name__ == "__main__":
    import numpy as np
    ai = AzureOpenAIEngine()
    
    # Test 1: CPI Shock
    res = ai.analyze_event("CPI m/m", actual=0.5, forecast=0.3)
    print(f"CPI Test: {res}")
    
    # Test 2: NFP Miss
    res = ai.analyze_event("Non-Farm Payrolls", actual=150, forecast=180)
    print(f"NFP Test: {res}")
