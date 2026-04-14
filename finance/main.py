from __future__ import annotations
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


def _safe_round(val, digits: int = 2):
    try:
        return round(float(val), digits) if val is not None else None
    except (TypeError, ValueError):
        return None


def _format_large(val):
    try:
        v = float(val)
        if v >= 1_000_000_000:
            return f"{v / 1_000_000_000:.2f}B"
        if v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        return str(round(v, 2))
    except (TypeError, ValueError):
        return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/quote/{ticker}")
async def get_quote(ticker: str) -> JSONResponse:
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info

        if not info or info.get("quoteType") is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

        result = {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
            "quote_type": info.get("quoteType"),
            "price": {
                "current": _safe_round(info.get("currentPrice") or info.get("regularMarketPrice")),
                "previous_close": _safe_round(info.get("previousClose")),
                "open": _safe_round(info.get("open")),
                "day_low": _safe_round(info.get("dayLow")),
                "day_high": _safe_round(info.get("dayHigh")),
                "week_52_low": _safe_round(info.get("fiftyTwoWeekLow")),
                "week_52_high": _safe_round(info.get("fiftyTwoWeekHigh")),
                "target_mean": _safe_round(info.get("targetMeanPrice")),
                "target_low": _safe_round(info.get("targetLowPrice")),
                "target_high": _safe_round(info.get("targetHighPrice")),
            },
            "valuation": {
                "market_cap": _format_large(info.get("marketCap")),
                "enterprise_value": _format_large(info.get("enterpriseValue")),
                "pe_trailing": _safe_round(info.get("trailingPE")),
                "pe_forward": _safe_round(info.get("forwardPE")),
                "peg_ratio": _safe_round(info.get("pegRatio")),
                "price_to_book": _safe_round(info.get("priceToBook")),
                "price_to_sales": _safe_round(info.get("priceToSalesTrailing12Months")),
                "ev_to_ebitda": _safe_round(info.get("enterpriseToEbitda")),
                "ev_to_revenue": _safe_round(info.get("enterpriseToRevenue")),
            },
            "financials": {
                "revenue_ttm": _format_large(info.get("totalRevenue")),
                "revenue_growth": _safe_round(info.get("revenueGrowth"), 4),
                "gross_margins": _safe_round(info.get("grossMargins"), 4),
                "operating_margins": _safe_round(info.get("operatingMargins"), 4),
                "profit_margins": _safe_round(info.get("profitMargins"), 4),
                "ebitda": _format_large(info.get("ebitda")),
                "free_cashflow": _format_large(info.get("freeCashflow")),
                "earnings_growth": _safe_round(info.get("earningsGrowth"), 4),
                "earnings_quarterly_growth": _safe_round(info.get("earningsQuarterlyGrowth"), 4),
            },
            "balance_sheet": {
                "total_cash": _format_large(info.get("totalCash")),
                "total_debt": _format_large(info.get("totalDebt")),
                "debt_to_equity": _safe_round(info.get("debtToEquity")),
                "current_ratio": _safe_round(info.get("currentRatio")),
                "quick_ratio": _safe_round(info.get("quickRatio")),
                "return_on_equity": _safe_round(info.get("returnOnEquity"), 4),
                "return_on_assets": _safe_round(info.get("returnOnAssets"), 4),
            },
            "analyst": {
                "recommendation": info.get("recommendationKey"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                "target_mean_price": _safe_round(info.get("targetMeanPrice")),
            },
            "dividend": {
                "rate": _safe_round(info.get("dividendRate")),
                "yield": _safe_round(info.get("dividendYield"), 4),
                "payout_ratio": _safe_round(info.get("payoutRatio"), 4),
            },
            "retrieved_at": datetime.utcnow().isoformat() + "Z",
        }

        logger.info("Quote fetched for %s: price=%s", ticker.upper(), result["price"]["current"])
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching quote for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quote/{ticker}/summary")
async def get_summary(ticker: str) -> JSONResponse:
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info

        if not info or info.get("quoteType") is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        currency = info.get("currency", "")
        name = info.get("longName") or info.get("shortName") or ticker

        lines = [
            f"**{name} ({ticker.upper()})** — Stand: {datetime.utcnow().strftime('%Y-%m-%d')} (Quelle: Yahoo Finance)",
            "",
        ]

        if price:
            lines.append(f"- Aktueller Kurs: {price} {currency}")
        if info.get("previousClose"):
            lines.append(f"- Vortageskurs: {info['previousClose']} {currency}")
        if info.get("fiftyTwoWeekLow") and info.get("fiftyTwoWeekHigh"):
            lines.append(f"- 52-Wochen-Range: {info['fiftyTwoWeekLow']} – {info['fiftyTwoWeekHigh']} {currency}")
        if info.get("marketCap"):
            lines.append(f"- Marktkapitalisierung: {_format_large(info['marketCap'])} {currency}")
        if info.get("trailingPE"):
            lines.append(f"- KGV (trailing): {_safe_round(info['trailingPE'])}")
        if info.get("forwardPE"):
            lines.append(f"- KGV (forward): {_safe_round(info['forwardPE'])}")
        if info.get("priceToBook"):
            lines.append(f"- Kurs-Buchwert-Verhältnis: {_safe_round(info['priceToBook'])}")
        if info.get("totalRevenue"):
            lines.append(f"- Umsatz (TTM): {_format_large(info['totalRevenue'])} {currency}")
        if info.get("profitMargins"):
            lines.append(f"- Nettomarge: {round(info['profitMargins'] * 100, 1)}%")
        if info.get("grossMargins"):
            lines.append(f"- Bruttomarge: {round(info['grossMargins'] * 100, 1)}%")
        if info.get("totalDebt"):
            lines.append(f"- Gesamtverschuldung: {_format_large(info['totalDebt'])} {currency}")
        if info.get("totalCash"):
            lines.append(f"- Liquide Mittel: {_format_large(info['totalCash'])} {currency}")
        if info.get("debtToEquity"):
            lines.append(f"- Verschuldungsgrad: {_safe_round(info['debtToEquity'])}")
        if info.get("returnOnEquity"):
            lines.append(f"- Eigenkapitalrendite: {round(info['returnOnEquity'] * 100, 1)}%")
        if info.get("dividendYield"):
            lines.append(f"- Dividendenrendite: {round(info['dividendYield'] * 100, 2)}%")
        if info.get("recommendationKey"):
            lines.append(f"- Analysten-Konsens: {info['recommendationKey']} ({info.get('numberOfAnalystOpinions', '?')} Analysten)")
        if info.get("targetMeanPrice"):
            lines.append(f"- Durchschnittliches Kursziel: {info['targetMeanPrice']} {currency}")

        return JSONResponse({"ticker": ticker.upper(), "summary": "\n".join(lines)})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching summary for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))
