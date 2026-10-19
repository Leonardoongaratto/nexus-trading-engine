# Nexus Trading Engine

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/Leonardoongaratto/nexus-trading-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Leonardoongaratto/nexus-trading-engine/actions/workflows/ci.yml)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checker: Mypy](https://img.shields.io/badge/type%20checker-mypy%20strict-blue.svg)](https://mypy-lang.org/)
[![Platform](https://img.shields.io/badge/broker-MetaTrader%205-orange.svg)](https://www.metatrader5.com/)

**Nexus Trading Engine** is a high-performance, deterministic multi-asset algorithmic trading engine designed for institutional risk management, market structure execution, and direct MetaTrader 5 connectivity.

Built with a hexagonal architecture and pure-function domain models, Nexus combines institutional market liquidity concepts with strict mathematical capital preservation rules.

---

## 🌟 Key Highlights

- **Multi-Asset Simultaneous Execution:** Concurrently monitors and trades Cryptocurrencies, Forex Majors, Precious Metals, and US Equity Indices with isolated risk allocations and independent magic numbers.
- **Institutional Risk Preservation:** Strictly guarantees maximum 1% risk per trade via dynamic lot sizing calibrated by account balance, leverage, tick values, spreads, and commissions.
- **Runner Trailing Strategy:** Automatically closes a 50% partial position at 1R, transitions the stop loss to Break-Even ($0.0R$), and clears the Take Profit to ride macro trends using a dynamic $2 \times \text{ATR}$ Trailing Stop.
- **Async MetaTrader 5 Engine:** Non-blocking order execution with automatic exponential backoff reconnection, jitter handling, and order verification.
- **Full Observability & Control:** Real-time HTTP REST endpoints (`/health`, `/status`), background watchdog health monitoring, and interactive Telegram bot integration.
- **Optional AI Guardian (LLM Veto):** Integrated contextual analysis to veto trade proposals before execution during hostile or anomalous macroeconomic conditions.

---

## 📊 Supported Asset Classes

| Asset Class | Symbols (Exness Standard / MT5) | Trading Schedule | Special Features |
| :--- | :--- | :--- | :--- |
| **Cryptocurrencies** | `BTCUSDm`, `ETHUSDm`, `SOLUSDm` | **24/7 Continuous** | Operates weekends & nights without session barriers |
| **Precious Metals** | `XAUUSDm` (Gold) | Institutional Sessions | London Open, NY Open, Asian Session |
| **Forex Majors** | `EURUSDm`, `GBPUSDm`, `USDJPYm` | London & NY Sessions | Ultra-tight spreads (0.2 - 0.3 pips) |
| **US Equity Indices** | `USTECm` (Nasdaq 100), `US30m` (Dow Jones) | NY Stock Exchange | High-volatility institutional trend surfing |

---

## 🏛️ System Architecture

```
                                  ┌──────────────────────────┐
                                  │   Market Feeds & Ticks   │
                                  └─────────────┬────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 NEXUS TRADING ENGINE                                   │
│                                                                                        │
│  ┌──────────────────────┐   ┌──────────────────────┐   ┌────────────────────────────┐  │
│  │   Market Structure   │   │   Risk Management    │   │     Session & Filters      │  │
│  │   - Order Blocks     │   │   - 1% Max Risk Sizing │   │   - Institutional Killzone │  │
│  │   - Fair Value Gaps  │──►│   - 1R 50% Partial   ├──►│   - Max Spread Filter      │  │
│  │   - BOS / ChoCH      │   │   - Break-Even Lock  │   │   - Economic News Calendar │  │
│  │   - Displacement     │   │   - 2x ATR Trailing  │   │   - 24/7 Crypto Mode       │  │
│  └──────────────────────┘   └──────────────────────┘   └────────────────────────────┘  │
│                                                │                                       │
│                                                ▼                                       │
│                            ┌───────────────────────────────────────┐                   │
│                            │   Order Executor & Proposal Manager   │                   │
│                            └───────────────────┬───────────────────┘                   │
└────────────────────────────────────────────────┼───────────────────────────────────────┘
                                                 │
                                                 ▼
                         ┌───────────────────────────────────────────────┐
                         │               Execution Ports                 │
                         ├───────────────────────┬───────────────────────┤
                         │    MetaTrader 5       │    Simulation Engine  │
                         │    (Live / Demo)      │    (Backtesting)      │
                         └───────────────────────┴───────────────────────┘
```

---

## 🛡️ Risk Management & Exit Architecture

1. **Deterministic Position Sizing:**
   $$\text{Lot Size} = \frac{\text{Account Balance} \times \text{Risk \%}}{(\text{Stop Distance} \times \text{Tick Value}) + \text{Spread Cost} + \text{Commission}}$$
   No trade is ever placed without a protective Stop Loss.
2. **Break-Even & Runner Execution:**
   - At $+1R$ profit: **50% of position volume is closed**.
   - Stop Loss is moved to **Entry Price (Break-Even)** to eliminate residual downside risk.
   - Fixed Take Profit is cleared (`take_profit = 0.0`), allowing the remaining position to surf trends indefinitely with an active Trailing Stop ($2 \times \text{ATR}$).

---

## 🚀 Quickstart

### Prerequisites
- Python 3.12+
- MetaTrader 5 Terminal installed (Windows)
- A broker account (Demo or Live, e.g. Exness)

### Installation

```bash
# Clone the repository
git clone https://github.com/Leonardoongaratto/nexus-trading-engine.git
cd nexus-trading-engine

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows

# Install in development mode
pip install -e ".[dev]"
```

### Configuration

Copy `.env.example` to `.env` and fill in your MT5 credentials:

```ini
XAU_ENV=development
XAU_BROKER_TYPE=mt5
XAU_MT5_LOGIN=your_account_login
XAU_MT5_PASSWORD=your_account_password
XAU_MT5_SERVER=your_broker_server

# Risk settings
XAU_RISK_PER_TRADE_PCT=1.0
XAU_EXECUTION_MODE=auto
```

### Running Nexus Engine

#### Multi-Asset Portfolio (Crypto + Metals + Forex + Indices):
```powershell
python -m xau_bot.cli run --broker mt5 --mode auto --timeframe-entry M1,M5 --symbol BTCUSDm,XAUUSDm,ETHUSDm,EURUSDm,GBPUSDm,USTECm,GBPJPYm
```

#### High-Frequency 24/7 Crypto Portfolio:
```powershell
python -m xau_bot.cli run --broker mt5 --mode auto --timeframe-entry M1,M5 --symbol BTCUSDm,ETHUSDm
```

---

## 📡 Live Observability

While running, Nexus exposes a lightweight HTTP monitoring server:

- **Health Check:** `curl http://127.0.0.1:8000/health`
  ```json
  {"status": "ok", "broker_connected": true, "last_tick_age_s": 0.42, "trading_halted": false}
  ```
- **Portfolio Metrics:** `curl http://127.0.0.1:8000/status`
  ```json
  {"broker_connected": true, "trading_halted": false, "positions": [], "execution_mode": "auto", "daily_pnl": 0.0}
  ```

---

## 🧪 Quality Gates & Testing

Nexus adheres to strict deterministic software engineering standards:

```powershell
# Format check
ruff format --check .

# Linting
ruff check .

# Static type checking
mypy src tests

# Automated test suite with coverage enforcement (min 80%)
pytest --cov=src --cov-fail-under=80
```

- **Current Suite:** 1,090+ tests, run on every push by the CI workflow.
- **Coverage:** ~95% across all modules, enforced at a minimum of 80%.

---

## ⚖️ Disclaimer

Trading foreign exchange, precious metals, indices, and cryptocurrencies carries a high level of risk and may not be suitable for all investors. This software is provided for educational, research, and algorithmic automation purposes. Always test on demo environments before allocating live capital.

---

## 📄 License

Copyright © 2026 Leonardo Custodio Ongaratto. All rights reserved.

The source code is published for portfolio and study purposes. No license is granted to copy, modify or use it in trading without written permission.
