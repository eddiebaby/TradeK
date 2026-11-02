# PRD: RSA Autopilot: Reverse Split Arbitrage Scout & Executor

| Field | Value |
|---|---|
| Owner | Scott Schweizer |
| Status | Draft v0.1, for review |
| Date | 2026-11-02 |
| Repo location | `TradeK/` (proposed code root: `src/rsa/`) |
| Parts | **Part 1: Scout** (find round-up deals) · **Part 2: Executor** (place 1-share orders across many brokerage accounts) |

Items marked **(verify)** are facts I couldn't confirm against a primary source while drafting. Check them before the build relies on them.

---

## 0. TL;DR

Today the workflow is: watch Reddit, newsletters and filings, read each filing for the fractional-share wording, then log into many brokerages and buy one share in each account by hand. Most deals don't work out, so most of that time is wasted.

RSA Autopilot splits the job in two:

1. **Scout** (no credentials, can run anywhere) does the following:
   - Continuously sweeps primary sources: SEC EDGAR full-text and latest-filings feeds, exchange and FINRA corporate-action lists, and press-release wires.
   - Sweeps secondary sources too: Reddit, newsletters, split calendars, and tracker sites.
   - Pulls the actual filing text and classifies the fractional-share treatment, with a **verbatim quote as evidence**.
   - Computes the buy-by deadline and expected profit per account, then pushes one alert or digest you can approve with a tap.
2. **Executor** (holds credentials, runs only on your own machine) does the following:
   - Fans one approved deal out to every enabled account.
   - Uses official broker APIs where they exist, scripted Playwright browser flows where they don't, and an AI-driven browser (Playwright MCP) only as a supervised fallback.
   - Verifies every fill, watches for the round-up shares to land, sells them, and records the outcome.
3. An **outcomes ledger** learns which brokers actually pass round-ups through for which filing wording. That cuts the "a lot of them don't work out" waste over time.

The goal is under 5 minutes of your time per trading day, with no duplicate or out-of-policy orders.

---

## 1. Background

### 1.1 How the trade works
A company does a 1-for-R reverse split. If you hold fewer than R shares, you'd end up with a fractional share. If the company **rounds fractions up to a whole share**, and your broker passes that through to you, one pre-split share becomes one post-split share worth roughly R times as much.

> Example: the stock is $0.40 and the split is 1-for-25. You buy 1 share for $0.40. After the split you'd hold 0.04 shares, which is rounded up to 1 share worth about $10.00. That's about $9.60 per account before post-split drift and fees. Across 15 accounts it's about $144.

**Round-lot protection** deals are a rarer, larger variant. Holders of at least 100 pre-split shares are topped up to 100 post-split shares. American Rebel (AREB) did this on its Feb 2, 2026 1-for-20 split and its Mar 23, 2026 1-for-100 split. The top-up is processed through DTC via broker election notices. That makes the deal much larger but much more broker-dependent.

### 1.2 What changed in 2025–2026 (and why this needs to be smarter than a keyword search)

| Change | Impact on the product |
|---|---|
| **Industry push against round-ups.** SIFMA (the broker-dealer trade group) published a Dec 2025 white paper and issuer best-practices guide that call out "round-up" exploitation and urge issuers toward cash-in-lieu. | Expect fewer clean round-ups. Every "round up" match needs its fine print read. The ledger must quantify whether the strategy is still worth running (see the §12 kill criterion). |
| **"Participant level" / "registered holder level" wording.** Beyond Meat, Auddia and Nexera (effective Nov 6, 2026) round up "at the DTC participant level". LQR House (Jul 2026) rounds up "at the registered holder level". | Under this wording the round-up goes to the *broker's aggregate position*. Whether *your* account gets a share is up to the broker. The classifier must split these out from holder-level round-ups and score them by broker pass-through history. |
| **Nasdaq mandatory halt before reverse splits.** Rule 4120(a)(14), now renumbered (a)(15), halts trading before post-market closes on the day before the market effective date. SR-NASDAQ-2026-057 (Jul 2026) extends the halt framework as Nasdaq moves toward 23/5 trading. | It creates a hard buy-by deadline, and the deadline can move as 23/5 trading rolls out. Deadline logic must be a configurable rule, not a constant. |
| **Short notice windows.** Nasdaq-listed issuers must notify Nasdaq ~10 calendar days ahead and disclose publicly at least 2 business days before the market effective date (Rule 5250; **verify** exact current text). | Sometimes only 2–3 trading days separate the announcement from the deadline. Scout must poll intraday, not once a day. |
| **More reverse splits coming.** NYSE's change approved Aug 2026, effective Jul 1, 2027, starts immediate delisting proceedings when a stock closes under $0.25. | Deal flow is likely to increase into 2027. |

### 1.3 Prior art
| Project/service | What it is | Use to us |
|---|---|---|
| **auto-rsa** (`NelsonDane/auto-rsa`, GPL-3.0) | CLI + Discord bot that buys, sells and checks holdings across ~16 brokers. Uses official APIs, then unofficial APIs, then Playwright/Selenium as a last resort. | It's the fastest path to multi-broker execution. Evaluate it in M0 as an execution backend (see §6.1). The author is not responding to issues, so treat it as a fork-and-own dependency. GPL is fine for personal, undistributed use. |
| `itsjafer/reverse-stock-split-arbitrage` | Older Python module (Robinhood/Alpaca/Webull) | Reference only |
| Idle Markets (Whop), tradelabs.org RSA signals, reversesplit.substack.com, New Moon Special Situations (Substack), stocksplitarbitrage.com | Paid or free alert services and newsletters | Secondary sources for Scout. Benchmarks for Scout recall. |
| Dilution Tracker reverse-split list, StockTitan split feed, TipRanks / MarketBeat / Capedge split calendars | Split calendars, some with float | Secondary corroboration + enrichment |
| Robinhood corporate-actions tracker | Shows per-split treatment (retained / cash in lieu / rounded up) after the fact | Ground truth for the outcomes ledger (Robinhood) |

---

## 2. Goals and non-goals

### Goals
- **G1: Coverage.** Detect every US-listed and OTC reverse split with favorable fractional treatment, early enough to act. That means before both the community trackers and the buy-by deadline.
- **G2: Accuracy.** Every alert carries the exact filing sentence that justifies it, plus a link. No unsupported "round up" claims.
- **G3: Hands-off execution.** A single approval places all orders across all enabled accounts, verifies them, and handles the sell side.
- **G4: Learn.** Track per-broker, per-wording outcomes, so the system stops recommending deals or brokers that don't pay.
- **G5: Safety.** Nothing outside policy can ever execute: no wrong ticker, no wrong quantity, no duplicate order, no runaway spend.

### Non-goals (v1)
- Market-timing or holding the stock for upside. v1 does the round-up capture only.
- Opening accounts, moving money between accounts, or any non-trade account action.
- Defeating CAPTCHAs or broker bot detection. When a broker challenges, the system hands off to you (§6.6).
- Operating accounts you don't own, unless you have documented authorization from the account holder (§11).
- Providing tax or legal advice. The system only produces records your tax preparer can use.

---

## 3. User stories
1. *As Scott, I get one morning digest (and real-time pings for urgent deals) listing actionable deals, each with: ticker, ratio, buy-by time, expected $/account, a confidence label, and the quoted filing sentence.*
2. *I tap **Approve** and every enabled account buys the right quantity at a capped limit price. I get a summary showing which accounts filled, which failed and why, and which need my MFA code.*
3. *When the post-split shares land, I get a "landed in 11/14 accounts" message and they're sold under my sell policy (auto or one tap).*
4. *Each month I see net P&L by broker and by filing-wording type, and a CSV for taxes.*
5. *When a broker breaks (UI change, expired session, MFA), I get a clear message with a screenshot, and the other brokers aren't blocked.*
6. *I can set "auto-execute deals with HIGH confidence and cost < $X" so I don't have to approve routine ones.*

---

## 4. Success metrics
| Metric | Target |
|---|---|
| Scout recall vs. union of community trackers + manual audit (weekly) | ≥ 95% of favorable deals found before buy-by |
| Scout precision (alerts labeled ROUND_UP_* that the filing text supports) | ≥ 95% |
| Detection latency from EDGAR publication | Median ≤ 15 min, p95 ≤ 60 min during market hours |
| Execution success (approved account-orders placed without manual help, excluding MFA prompts) | ≥ 95% for API brokers, ≥ 85% for browser brokers |
| Duplicate orders / orders outside policy | **0** (hard requirement) |
| Your hands-on time | ≤ 5 min per trading day on average |
| Economics | Net P&L/month tracked; strategy-level kill criterion in §12 |

---

## 5. Part 1: Scout (discovery)

### 5.1 Source inventory

**Tier 1: primary, authoritative (a deal can be auto-executed only if a Tier-1 source confirms it)**

| Source | What it gives | Access | Cadence |
|---|---|---|---|
| **SEC EDGAR full-text search** (`efts.sec.gov/LATEST/search-index`, params `q`, `forms`, `dateRange/startdt/enddt`, `from`) | Phrase search across 8-K, 6-K, PRE/DEF 14A, PRE/DEF 14C, DEFA14A, 10-Q/10-K | HTTPS JSON. Declared `User-Agent` with contact email required; ≤ 10 req/s SEC fair-access limit | Every 10 min during 6 a.m.–8 p.m. ET, hourly overnight |
| **EDGAR latest-filings Atom feeds** (by form type) + filing index pages | Near-real-time new filings; covers full-text index lag | HTTPS Atom/HTML | Every 2–5 min during market hours |
| **8-K exhibits**: EX-99.1 (press release), EX-3.1 (charter amendment) | Final ratio, effective date, CUSIP, treatment language | Fetched from the filing index | On detection |
| **8-K Item 5.07** (vote results) | Shareholder approval of the split authority | Same | On detection |
| **Nasdaq Trader daily list / corporate actions** | Market effective date, new CUSIP for Nasdaq listings | HTTPS download | Every 30 min |
| **NYSE / NYSE American corporate-action notices** | Same for NYSE listings | HTTPS | Every 30 min |
| **FINRA Daily List / OTC corporate actions** | OTC reverse splits (where EDGAR often lags or is absent) | HTTPS / FINRA API (**verify** current access) | Every 30 min |
| **Press-release wires**: GlobeNewswire, PR Newswire, Business Wire, ACCESSWIRE, Newsfile | Announcements that often precede or accompany the 8-K | RSS/search pages with `"reverse stock split"` | Every 5 min during market hours |

**Tier 2: community and aggregators (discovery + corroboration + broker-outcome reports; never sufficient alone)**

| Source | Access notes |
|---|---|
| **Reddit RSA communities** (confirm the exact subreddit names you follow; also r/pennystocks and ticker subs for mentions) | Reddit's Responsible Builder Policy (Nov 2025) requires approval for API access. Reports say RSS is being shut down mid-Nov 2026 (**verify**). Build a pluggable connector with three modes: approved API app → logged-in browser read of your own account at low frequency → manual paste/forward. The most valuable Reddit content is **broker outcome reports** ("Fidelity rounded up XYZ"), which feed the ledger priors. |
| **Newsletters** (reversesplit.substack.com, New Moon Special Situations, others you subscribe to) | Ingest from a dedicated email inbox via IMAP filter. No scraping of paywalled pages. |
| **Split calendars**: Dilution Tracker, StockTitan, TipRanks, MarketBeat, Capedge, stocksplitarbitrage.com, tradelabs.org | Polite HTTP fetch where the site's terms allow; otherwise manual. Use for cross-checking recall. |
| **Discord RSA servers** | Self-bots (user-token scraping) violate Discord ToS, so they're excluded. Supported: a bot in a server you own, plus manual forwarding into a `#rsa-inbox` channel. |
| **Broker corporate-action pages** (Robinhood tracker, etc.) | Ground truth for outcomes, not discovery |
| X/StockTwits | Optional, low priority |

**Tier 3 (later):** transfer-agent notices (Continental, VStock, Equiniti, Computershare), ADR depositary notices (usually cash-in-lieu), ETF issuer notices (usually cash redemption of fractions).

### 5.2 Pipeline

```
[Collectors] ──► raw_items ──► [Candidate filter] ──► [Doc fetch + section extract]
   (per source)                  regex/keyword          primary doc + EX-99.1/EX-3.1
                                                                │
                                                                ▼
[Alert/Digest] ◄── [Scoring] ◄── [Enrichment] ◄── [Dedupe/merge into Deal] ◄── [Extractor + Classifier]
  Discord/Telegram   EV/account    price, float,     key = CIK (or symbol)        rules first, then LLM
  w/ Approve btn     deadline      ratio, dates      + split event                structured JSON + verbatim quote
```

1. **Collect.** Each source collector emits `raw_item {source, url, fetched_at, title, text, published_at}`. Collectors are idempotent: a content hash prevents re-processing.
2. **Candidate filter.** This is cheap regex. It keeps items that mention `reverse (stock )?split|share consolidation|combination of shares` and any of `fraction|round|lieu|round lot`.
3. **Fetch and section-extract.** For EDGAR, pull the primary doc and EX-99.1/EX-3.1, convert HTML to text, and keep windows of about 1,500 chars around each fractional-share mention.
4. **Extract and classify.** Rules run first; the regex library is in Appendix B. An LLM (Claude via the Anthropic API) then fills the §5.4 schema from the sections only.
   - The LLM must return verbatim evidence quotes.
   - A deterministic check confirms each quote appears in the source text. If it doesn't, `treatment = UNKNOWN`.
   - LLM output is treated as data and never as instructions (see §7).
5. **Dedupe and merge.** Many items describe one split event. The merge key is `(CIK or symbol, split event)`. A proxy's ratio *range* is later replaced by the 8-K/PR's *final* ratio. Every source stays attached as provenance.
6. **Enrich.**
   - Price (last, bid/ask, avg volume) comes from the Schwab market-data API (OAuth already exists in `src/schwab/`), with a fallback provider.
   - Also added: exchange/OTC, float, the trading calendar, and the Nasdaq/NYSE corporate-action record.
7. **Score** (§5.6) and **alert** (§5.7).

### 5.3 Fractional-treatment taxonomy

| Code | Typical wording | Default verdict |
|---|---|---|
| `ROUND_UP_HOLDER` | "fractional shares … will be rounded up to the nearest whole share", with no level qualifier, or explicitly per stockholder or beneficial owner | **Go** |
| `ROUND_LOT_PROTECTION` | "holders of at least 100 shares … will receive no fewer than 100 shares" / "round lot" | **Go, manual approval always** (larger size, broker election dependency) |
| `ROUND_UP_PARTICIPANT` | "rounded up … at the DTC participant level" | **Broker-dependent** (scored per broker by ledger history) |
| `ROUND_UP_REGISTERED` | "rounded up at the registered/record holder level" | **Broker-dependent** (street-name shares are held of record by Cede & Co.) |
| `ROUND_UP_UNSPECIFIED` | Round-up stated, but the mechanics are ambiguous or carry board-discretion caveats | Go-with-caution |
| `CASH_IN_LIEU` | "cash in lieu", "cash payment" | No |
| `AGGREGATE_AND_SELL` | Fractions aggregated and sold, proceeds distributed | No |
| `ROUND_DOWN` / `ISSUE_FRACTIONAL` | Fractions cancelled or issued as fractions | No |
| `UNKNOWN` | Not stated, or the extractor couldn't verify a quote | Hold; re-check when the 8-K or PR lands |

The classifier also flags **modifiers**:
- An anti-arbitrage clause (e.g., the issuer reserves the right to refuse round-ups for positions acquired after a date).
- A settled-shares requirement.
- A record date that differs from the effective date.
- The ratio being a range at the board's discretion.
- A split that has been withdrawn or delayed.

### 5.4 Deal record (extraction schema)

```json
{
  "deal_id": "uuid",
  "issuer": {"name": "...", "cik": "...", "symbol": "XYZ", "exchange": "NASDAQ|NYSE|NYSE_AMERICAN|OTC", "cusip_old": "...", "cusip_new": "..."},
  "status": "PROPOSED|APPROVED|ANNOUNCED|EFFECTIVE|COMPLETED|WITHDRAWN",
  "ratio": {"final": 25, "range_min": 5, "range_max": 50},
  "dates": {"meeting": "...", "record": "...", "effective": "...", "market_effective": "2026-11-06", "buy_by": "2026-11-05T15:55:00-05:00"},
  "treatment": "ROUND_UP_PARTICIPANT",
  "modifiers": ["SETTLED_SHARES_REQUIRED"],
  "evidence": [{"quote": "…rounded up to the nearest whole share at the DTC participant level…", "url": "https://www.sec.gov/Archives/...", "form": "8-K", "filed_at": "..."}],
  "sources": [{"tier": 1, "kind": "edgar_8k", "url": "..."}, {"tier": 2, "kind": "reddit", "url": "..."}],
  "confidence": "HIGH|MEDIUM|LOW",
  "economics": {"px_pre": 0.40, "qty_pre": 1, "qty_post": 1, "gross_per_acct": 9.60, "ev_per_acct_by_broker": {"fidelity": 7.10, "schwab": 6.50}},
  "needs_human": false
}
```

### 5.5 Deal lifecycle

`PROPOSED` (PRE/DEF 14A/14C with a ratio range) → `APPROVED` (8-K Item 5.07 or written consent) → `ANNOUNCED` (8-K/PR with the final ratio and effective date) → `EFFECTIVE` (market effective date) → `COMPLETED` (ledger closed). It can also move to `WITHDRAWN` at any point.

- Proxies are tracked as a **watchlist**. They're often the earliest signal, sometimes weeks ahead, and give time to pre-check price and liquidity.
- Alerts with an Approve button only fire at `ANNOUNCED`, unless you enable early-entry.

### 5.6 Economics and scoring

```
R          = final ratio (or range_min while unknown; conservative)
qty_pre    = 1 for round-up; 100 for ROUND_LOT_PROTECTION
qty_post   = ceil(qty_pre / R) → 1;  ROUND_LOT: max(ceil(qty_pre/R), 100)
px_post    = px_pre × R × (1 − haircut)        # haircut default 0.25 (post-split decline), tunable from ledger
gain       = qty_post × px_post − qty_pre × px_ask − fees(broker)
p_pass     = P(round-up received | broker, treatment, exchange)   # from ledger; priors until enough data
EV(broker) = p_pass × gain + (1 − p_pass) × (−qty_pre × px_ask × haircut) − fees(broker)
```

- **Deal score** = Σ EV over enabled accounts, rank-ordered.
- **Confidence** = f(Tier-1 corroboration, quote verified, treatment class, modifiers, data freshness).
- **Filters** (configurable): EV per account ≥ $1.00; px_ask ≤ $X; no OTC Expert Market (can't buy); spread ≤ Y%.

### 5.7 Deadlines and alerting
- **Default buy-by** = the 4:00 p.m. ET regular-session close on the trading day before `market_effective`.
  - Optional: extend to just before the exchange's pre-split halt (Nasdaq: before post-market closes) for brokers that support extended-hours orders.
  - `SETTLED_SHARES_REQUIRED` moves buy-by one more trading day earlier (T+1 settlement).
  - Deadline rules live in versioned config (Nasdaq 23/5 trading will change them).
- **Alert tiers**:
  - Urgent: < 24h to buy-by, sent in real time.
  - Normal: included in the morning digest at 8:30 a.m. ET.
  - Watchlist: weekly.
- **Digest content** (per deal):
  - Ticker, ratio, buy-by countdown, treatment code + quote, EV per account and in total, confidence.
  - Buttons: Approve, Approve (custom accounts), Skip, Snooze.
- **Change alerts**: ratio finalized, effective date moved, withdrawal, treatment reclassified after a new filing.

### 5.8 Scout functional requirements
| ID | Requirement |
|---|---|
| FR-S1 | Collectors for every Tier-1 source in §5.1, each with health checks and "last successful fetch" tracking. A silent source raises an alert after 2× its cadence. |
| FR-S2 | The EDGAR client sends a declared User-Agent, stays ≤ 5 req/s (half the SEC limit), and retries with backoff on 403/429. Query set in Appendix A. |
| FR-S3 | Rules + LLM extraction to the §5.4 schema, with verbatim-quote verification. Any unverified field is set to null and marked `needs_human`. |
| FR-S4 | Dedupe/merge across sources and over the deal lifecycle, with full provenance. |
| FR-S5 | Price/liquidity enrichment including OTC tiers. Skip deals that can't be bought (Expert Market, halted). |
| FR-S6 | Scoring with per-broker EV using ledger priors (§8). |
| FR-S7 | Real-time alert + daily digest with inline approval actions (Discord bot by default; Telegram/ntfy optional). |
| FR-S8 | Backfill mode: run over the last 24 months of filings to build the labeled corpus (§10) and seed ledger priors from public outcome reports. |
| FR-S9 | Scout holds **no brokerage credentials** and can run in the cloud, on a schedule, or locally. |

---

## 6. Part 2: Executor (multi-broker order placement)

### 6.1 Approach: tiered adapters behind one interface

Every broker is reached through one `BrokerAdapter` contract. Each adapter uses the most deterministic method available:

1. **Official API** (preferred: fast, reliable, lowest ToS risk).
2. **Scripted Playwright** (deterministic browser flow with a persistent profile per login).
3. **Bridge to auto-rsa** (during M1–M3, for brokers it already supports and we haven't scripted).
4. **AI-driven browser fallback**, using the Playwright MCP server driven by Claude. Used only:
   - (a) to help author or repair a Playwright script, or
   - (b) to complete a single order when a scripted flow breaks.

   It always works under the two-phase commit in §6.4, and it never decides what to trade.

Why not AI-driven browsing for everything? For money-moving actions you want determinism, speed and auditability. LLM-driven clicking is slower, costs per run, and can be steered by page content. It's the right tool for self-healing and onboarding, not the default path.

```python
class BrokerAdapter(Protocol):
    broker_id: str
    method: Literal["api", "playwright", "autorsa_bridge", "ai_browser"]
    def ensure_session(self, login: LoginRef) -> SessionState            # may raise NeedsHuman(mfa|captcha|reauth)
    def list_accounts(self, login: LoginRef) -> list[AccountRef]
    def get_positions(self, acct: AccountRef) -> list[Position]
    def get_open_orders(self, acct: AccountRef) -> list[Order]
    def preview_order(self, acct: AccountRef, intent: OrderIntent) -> OrderPreview   # phase 1
    def submit_order(self, acct: AccountRef, preview: OrderPreview) -> OrderReceipt  # phase 2
    def get_order(self, acct: AccountRef, broker_order_id: str) -> Order
    def cancel_order(self, acct: AccountRef, broker_order_id: str) -> None
```

### 6.2 Initial broker matrix

Fill in the **Your accounts** column. This is the main open question in §14.

| Broker | Recommended method | Auth / MFA handling | Notes |
|---|---|---|---|
| Charles Schwab | Official Trader API (OAuth already in `src/schwab/`) | OAuth. Refresh token lasts 7 days, so a weekly browser re-auth prompt is scheduled | auto-rsa uses Playwright plus a TOTP secret instead. thinkorswim must be enabled per account for that path. |
| Public | Official Individual API (personal secret → short-lived tokens) | No interactive MFA | Personal, non-commercial use only |
| Tradier | Official API | Access token | |
| tastytrade | Official API (OAuth2) | Refresh token | |
| Webull | Official OpenAPI (application required) → fallback unofficial | App key / trading PIN | Approval status **(verify)** |
| E*TRADE | Official API (OAuth 1.0a) | Tokens expire daily, so a morning re-auth is needed | **(verify)** current program terms |
| Interactive Brokers | Official Client Portal/Web API or TWS | Gateway session, daily re-auth | Round-up pass-through **(verify via ledger)** |
| Fidelity | Scripted Playwright | TOTP secret | No public API |
| Robinhood | Scripted Playwright (or unofficial API) | TOTP secret | No official equities API |
| SoFi | Scripted Playwright | TOTP secret | |
| Firstrade | Scripted Playwright (or unofficial API) | PIN/OTP | |
| Vanguard | Scripted Playwright | SMS → human hand-off, or keep a long-lived session | Slow flows. Watch for fractional/odd-lot restrictions. |
| Chase (Self-Directed) | Scripted Playwright | SMS → human hand-off | Requires the "classic" trading UI per auto-rsa docs |
| Wells Fargo | Scripted Playwright | SMS → human hand-off | |
| Merrill Edge | Scripted Playwright | SMS/app | |
| Ally Invest | Scripted Playwright | | Per-trade fees reported **(verify)**; enter them into `fees(broker)` |
| BBAE, DSPAC, Fennel, Tornado | Via auto-rsa bridge | per auto-rsa | Small brokers used by the RSA community |

### 6.3 End-to-end workflow

```
Deal APPROVED (by Scott, or by an auto-approve rule)
  └─► Plan: for each enabled account → OrderIntent(symbol, BUY, qty, limit=min(ask×1.03, cap), TIF=DAY,
                                                   idempotency_key=hash(deal_id, account_id, "BUY"))
       └─► Pre-trade checks (deterministic, §7.2)
            └─► Fan out: brokers in parallel; accounts within one broker sequential, with 5–20 s jitter
                 └─► per account: ensure_session → check positions/open orders (idempotency)
                      → preview → verify preview == intent → submit → poll fill (≤ 10 min)
                      → unfilled? re-price once within cap, else cancel and report
  └─► Summary message: filled / failed (reason + screenshot) / needs-MFA (reply with code)
Post-split monitor (daily, from market_effective until landed or T+15 business days):
  └─► get_positions → classify per account: ROUNDED_UP | CASH_IN_LIEU | FRACTION_RETAINED | MISSING
       (positions often show 0 for 1–2 days before the round-up posts; don't mark MISSING until the grace period ends)
  └─► Sell policy: auto | one-tap | hold. Limit sell at bid (or marketable limit) for the landed shares only
  └─► Ledger entry finalized (§8)
```

### 6.4 Two-phase commit for any browser path
1. **Phase 1 (fill and preview).** The adapter (scripted or AI-driven) fills the order ticket and stops on the broker's **preview/review** screen.
2. **Verify.** Deterministic code reads the preview DOM/text and checks symbol, side, quantity, order type, limit price and account number against the `OrderIntent`. It saves a screenshot to the audit log.
3. **Phase 2 (submit).** Only on an exact match does the adapter click Submit and capture the confirmation number. Any mismatch means abort and alert.
4. The AI fallback has **no tool to change the intent**. It receives the intent as fixed parameters and can only navigate and fill. Its allowed URLs are limited to the broker's trade/positions pages, so it never touches transfers, banking or settings.

### 6.5 Automation modes (per deal class, configurable)
| Mode | Behavior | Default for |
|---|---|---|
| **Alert-only** | Scout alerts; you trade manually | `ROUND_UP_UNSPECIFIED`, `UNKNOWN` |
| **One-tap** (default) | Approve in Discord → full fan-out | `ROUND_UP_HOLDER`, `ROUND_UP_PARTICIPANT/REGISTERED` (only brokers with p_pass ≥ threshold are pre-selected) |
| **Auto-execute** | Executes without approval if: confidence HIGH, Tier-1 corroborated, total cost ≤ daily cap, EV/acct ≥ min | Opt-in, after M4 burn-in |
| **Manual only** | Never auto, always reviewed | `ROUND_LOT_PROTECTION`, anything with anti-arbitrage modifiers |

### 6.6 Sessions, MFA and human hand-off
- **Persistent browser profiles.** There's one Playwright user-data-dir per broker login, under an encrypted local folder. This mirrors Playwright MCP's persistent-profile mode (`--user-data-dir`), which preserves cookies and storage between runs. One browser instance per profile at a time, so the executor holds a per-profile lock.
- **TOTP secrets.** Where the broker supports authenticator apps, store the TOTP seed in the vault and generate codes locally. This removes most prompts.
- **SMS/push MFA.** The executor pauses that broker only, pings you ("Vanguard wants a code"), and accepts a reply in the bot thread. It times out after 10 minutes and skips that broker's accounts for this deal.
- **CAPTCHA / bot challenge / "unusual activity" page.** Stop that broker, screenshot it, and notify you. No automated solving or evasion.
- **Session warm-up.** About 30 minutes before the market open on days with pending deals, run `ensure_session` for all logins so MFA prompts surface early, not at 3:50 p.m.

### 6.7 Where it runs
- **Executor:** your own always-on machine (desktop or a mini-PC/home server), on your home internet connection.
  - Credentials and browser profiles never leave it.
  - Brokers tend to challenge datacenter IPs and fresh browser fingerprints.
  - Runs headed or headless Chromium with persistent profiles. Not this cloud container.
- **Scout:** anywhere (cloud cron, Docker on the same box). It talks to the Executor only through the approval queue (local DB/API). The Executor **pulls** approved deals; nothing inbound can trigger trades.

### 6.8 Executor functional requirements
| ID | Requirement |
|---|---|
| FR-X1 | `BrokerAdapter` contract + adapters per the §6.2 matrix. Each adapter declares its capabilities (extended hours, fractional, OTC, max price). |
| FR-X2 | Idempotent order placement. Before every submit (including retries), check existing positions and open orders for the deal's symbol and skip if already satisfied. |
| FR-X3 | Two-phase commit (§6.4) for every browser-based submit. |
| FR-X4 | Limit orders only, DAY TIF. Limit = min(ask × (1 + slippage), per-deal cap). One re-price attempt, then cancel. |
| FR-X5 | Per-broker sequential execution with jitter. Cross-broker parallelism (configurable max concurrency). |
| FR-X6 | MFA/CAPTCHA hand-off per §6.6. One broker failing never blocks the others. |
| FR-X7 | Post-split monitor with a grace period, per-account outcome classification, and a sell policy. |
| FR-X8 | Dry-run mode (everything except phase-2 submit) and paper mode (broker sandboxes where available: Tradier sandbox, tastytrade sandbox, IBKR paper, Webull paper). |
| FR-X9 | Kill switch: `rsa halt` CLI + bot command + a file flag. All workers check it before every submit. |
| FR-X10 | Full audit trail: every intent, preview, screenshot, submit, fill and error, append-only. |

---

## 7. Safety, security and guardrails

### 7.1 Secrets
- Credentials and TOTP seeds go in the OS keychain (`keyring`) or a 1Password/Bitwarden CLI vault, referenced by ID.
- Never in `.env` files committed to git, never in logs, never sent to an LLM.
- Browser profiles live in an encrypted directory (full-disk encryption at minimum).
- Add `rsa-data/`, `profiles/`, `creds/` and `*.har` to `.gitignore` before any code lands.
- Broker API tokens are refreshed and stored with the existing `src/schwab/token_manager.py` pattern (encrypted at rest).

### 7.2 Deterministic pre-trade checks (cannot be bypassed by config at runtime)
- The symbol must belong to a deal in `APPROVED` state with ≥ 1 verified Tier-1 evidence quote. **A Reddit- or Discord-only deal can never execute.**
- `qty == 1`, or `qty == 100` for `ROUND_LOT_PROTECTION`. Never anything else.
- Limit price ≤ per-deal cap. Per-order notional ≤ $Z. Daily total spend ≤ $D. Open RSA exposure ≤ $E.
- Now < buy-by. The account is enabled. The broker's health is green.
- Side SELL only for quantities ≤ the landed post-split position for that deal.

### 7.3 Prompt-injection posture
Scout feeds untrusted web text (Reddit, newsletters, PR wires) into an LLM. The defenses:
- LLM output is parsed as **data** into a strict schema. It can't call tools or trigger actions.
- Evidence quotes must string-match the source.
- The Executor never consumes LLM text. It consumes only validated `Deal` rows that passed §7.2.
- The AI browser fallback receives fixed parameters, is URL-allowlisted to the broker's trading pages, and is subject to the two-phase verify.

### 7.4 Operational safety
- Every outbound order is logged *before* submit (write-ahead).
- Crash recovery reconciles against broker open orders and positions before resuming.
- On any unexpected state (position larger than expected, unknown open order), halt that broker and alert.

---

## 8. Outcomes ledger and learning loop
- **Per (deal, account) row:**
  - bought? fill px/time; pre/post qty expected
  - post-split result (`ROUNDED_UP` / `CASH_IN_LIEU` / `FRACTION_RETAINED` / `MISSING`) and days-to-land
  - sell px/time, fees, realized P&L, notes
- **Aggregates:**
  - p_pass by broker × treatment × exchange (Beta-Binomial with community-report priors; the posterior feeds §5.6)
  - average post-split haircut by days-held
  - P&L by month/broker/treatment
- **Exports:**
  - Monthly CSV and per-broker CSVs for 1099-B reconciliation.
  - Cash-in-lieu is reported as a sale (IRS Pub. 550). Round-up shares carry the original basis, which your tax preparer will confirm.
- **Feedback into Scout:** brokers with low p_pass for a treatment class are auto-deselected for that class, which you can override.

---

## 9. Architecture

### 9.1 Components
```
┌──────────────────────── Scout (no creds) ────────────────────────┐
│ collectors/  edgar_fts, edgar_feed, nasdaq, nyse, finra, wires,   │
│              reddit, newsletter_imap, calendars                   │
│ extract/     sectionizer, rules, llm_extractor, quote_verifier    │
│ deals/       merge, lifecycle, enrich (prices via src/schwab),    │
│              scoring                                              │
│ notify/      discord_bot (digest + approve), telegram/ntfy        │
└──────────────┬───────────────────────────────────────────────────┘
               │  SQLite/Postgres: deals, approvals, ledger (single source of truth)
┌──────────────▼──────────── Executor (local only) ─────────────────┐
│ planner → risk_checks → dispatcher (per-broker queues, locks)     │
│ adapters/ api_{schwab,public,tradier,tastytrade,webull,etrade,ibkr}│
│           pw_{fidelity,robinhood,vanguard,chase,sofi,wf,merrill…} │
│           autorsa_bridge, ai_browser (Playwright MCP)             │
│ monitor/  post_split_watcher, sell_engine                         │
│ vault/    keyring / 1Password CLI · profiles/ (encrypted)         │
│ audit/    append-only log + screenshots                           │
└───────────────────────────────────────────────────────────────────┘
```

### 9.2 Tech stack (fits the existing repo)
- **Core:** Python 3.11+, Pydantic models, SQLAlchemy/SQLModel on SQLite (single user; Postgres optional). APScheduler for cadences. structlog.
- **HTTP and parsing:** httpx with retry/backoff; selectolax/BeautifulSoup for HTML.
- **Broker and market APIs:** Playwright for Python for scripted adapters. Playwright MCP server for the AI fallback and script authoring. Official SDKs where they exist; the existing `src/schwab/` OAuth client for Schwab trading and market data.
- **LLM:** Anthropic SDK, with a small, fast model for triage and a larger model for extraction. Model names live in config. Expected volume is tens of documents a day, so the cost is negligible.
- **Notifications:** discord.py bot for digests and approvals (`src/discord/` has only a test scaffold today).
- **Interfaces:** CLI via Typer: `rsa scout run`, `rsa deals`, `rsa approve`, `rsa exec --dry-run`, `rsa halt`, `rsa ledger export`.

### 9.3 Proposed repo layout
```
src/rsa/
  scout/{collectors,extract,deals,notify}/
  executor/{planner.py,risk.py,dispatcher.py,adapters/,monitor/,vault/}
  ledger/
  config/  (deadline_rules.yaml, brokers.yaml, policy.yaml; no secrets)
tests/rsa/
  fixtures/filings/   (labeled corpus)   fixtures/brokers/ (recorded DOM/API responses; scrubbed)
```

---

## 10. Testing strategy (per repo TDD conventions)
- **Labeled filing corpus.** At least 150 real filings from 2024–2026, covering every §5.3 class and including tricky modifiers. Each one is labeled once by you (or by a review pass). The classifier must hit ≥ 95% accuracy on Go/No-go, with 100% on "never call CASH_IN_LIEU a Go".
- **Unit tests** (fast, isolated):
  - regex library, quote verifier, merge/dedupe, deadline math, EV math
  - risk checks: a property test asserts that no generated `OrderIntent` can exceed caps or carry qty ∉ {1, 100}
- **Property tests** (Hypothesis): trading-calendar deadline computation across holidays, half-days, DST and 23/5 configs; idempotency under random retry and crash points.
- **Adapter contract tests.** Every adapter is run against recorded fixtures (scrubbed DOM snapshots and API responses), with the same suite for every adapter. Nightly smoke runs in **dry-run** against real brokers to detect UI drift.
- **Chaos tests:** session expiry mid-order, MFA prompt at submit, network drop after submit (must reconcile, not double-buy), preview mismatch (must abort).
- **E2E:** paper and sandbox brokers only. **No live orders in CI, ever.**
- **Quality gates:** coverage ≥ 90%; mutation score ≥ 80% on `risk.py`, `scoring.py`, `quote_verifier.py`, `deadline.py`.

---

## 11. Compliance, terms of service and tax (plain-language)
- **The strategy itself** is ordinary stock buying and selling. The edge comes from issuer-chosen fractional treatment. Industry groups (SIFMA) are actively discouraging round-ups, so expect the edge to shrink and wording to get stricter.
- **Account ownership.** Automate only accounts you own. For a family member's account, you need their documented authorization and the broker's rules permitting it. Never create accounts with inaccurate information.
- **Broker ToS.** Many brokers' agreements restrict automated or scripted access. Official APIs carry the lowest risk. Browser automation of your own login can lead to account restrictions or closure.
  - Mitigations: human-like pacing, no evasion tooling, low volume.
  - Decide per broker whether the convenience is worth the risk. The matrix in §6.2 lets you disable any broker.
- **Data sources.** Respect the SEC fair-access policy, Reddit's developer policy, Discord ToS (no self-bots) and site terms. Don't redistribute paid newsletter content.
- **Taxes.** Many small lots across many 1099-Bs. The ledger export exists so reconciliation is mechanical. Confirm treatment of round-up shares and cash-in-lieu with your tax preparer.

---

## 12. Risks and mitigations
| Risk | Likelihood | Mitigation |
|---|---|---|
| Issuers shift to cash-in-lieu or participant-level wording (SIFMA push) | High | Ledger quantifies yield. **Kill criterion:** if net P&L < $50/month for 3 consecutive months, pause Executor development and keep Scout in alert-only mode. |
| Broker keeps the participant-level round-up and gives you nothing | High | Per-broker p_pass from the ledger. Auto-deselect losers. |
| Broker UI change breaks a script on deal day | Medium | Nightly dry-run smoke tests, AI fallback with two-phase verify, per-broker isolation. |
| MFA prompts at the wrong time | Medium | TOTP seeds where possible, morning session warm-up, hand-off with timeout. |
| Duplicate orders from a retry or crash | Low (high impact) | Write-ahead log, idempotency keys, position/open-order reconciliation before every submit. |
| Missed deal (short notice, source outage) | Medium | Multiple Tier-1 sources, source-silence alerts, intraday cadence, weekly recall audit vs. trackers. |
| Wrong classification (e.g., misses "except" clause) | Medium | Verbatim quotes in alerts, a labeled corpus, conservative defaults (`UNKNOWN` = hold). |
| Prompt injection from scraped content | Low | §7.3 design: LLM output is data, Tier-1-only execution. |
| Account restriction for automation | Medium | Prefer official APIs, pacing, per-broker opt-in, no evasion. |
| Split withdrawn or delayed after you buy | Low–Med | Change alerts. Position is tiny (1 share); default policy is sell on withdrawal. |
| Reddit/RSS access shutdown | High | Reddit is Tier 2 only, with a pluggable connector and a manual-forward path. |

---

## 13. Milestones

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M0: Spike** | Run EDGAR FTS queries over the last 90 days and hand-check the results. Install and dry-run auto-rsa against 2–3 of your brokers. Confirm your broker list and the Reddit/newsletter sources. | A go/no-go on auto-rsa as a bridge. A query set tuned for recall. |
| **M1: Scout MVP (alert-only)** | EDGAR FTS + feed, wires, Nasdaq/NYSE lists. Rules + LLM extraction with quote verification. Discord digest. Labeled corpus v1. | ≥ 90% recall over 2 weeks vs. trackers; zero unsupported Go alerts. |
| **M2: Executor on API brokers (dry-run → live)** | Adapter contract, risk checks, idempotency, audit log, kill switch. Schwab, Public and Tradier (plus whichever API brokers you use). One-tap approve. | 2 weeks of dry-run with zero policy violations, then live on 1–2 deals. |
| **M3: Browser brokers** | Scripted Playwright for your browser-only brokers, two-phase commit, MFA hand-off, session warm-up. auto-rsa bridge for the rest. | ≥ 85% unattended success over 10 deals. |
| **M4: Post-split + sell + ledger** | Landing monitor, sell policy, outcomes ledger, tax export, p_pass feedback into scoring. | Every deal from M2+ fully reconciled. |
| **M5: Self-healing + auto mode** | Playwright MCP fallback for broken flows. Opt-in auto-execute for HIGH-confidence deals. Tier-2 collectors (Reddit, newsletters, calendars). | 30 days with < 5 min/day of your time. |

---

## 14. Open questions for Scott
1. **Which brokers, and how many accounts at each?** Which logins are yours and which are family members'? This decides the adapter build order.
2. **Approval channel:** Discord (matches the existing `src/discord/` scaffold), Telegram, or SMS/ntfy?
3. **Default automation mode** per treatment class (§6.5). Should auto-execute ever be on?
4. **Caps:** max price per share, max per-order notional, daily spend cap, total open exposure.
5. **Sell policy:** auto-sell on landing, one-tap, or hold N days?
6. **Sources:** exact subreddits, newsletters and Discords you follow today, and which are paid.
7. **Runtime host:** which machine runs the Executor (OS, always-on?), and which secrets manager you use.
8. **auto-rsa:** are you open to using it as a bridge (GPL-3.0, unmaintained issues), or do you want everything in-house?
9. **Round-lot-protection deals** (100-share buys): include them, with a per-deal manual approval?

---

## Appendix A: EDGAR full-text query set (starting point; tune in M0)

Forms: `8-K,6-K,PRE 14A,DEF 14A,DEFA14A,PRE 14C,DEF 14C`, plus `10-Q,10-K` for the backfill.

| # | Query (`q=`) | Purpose |
|---|---|---|
| 1 | `"reverse stock split" "rounded up"` | Core positive |
| 2 | `"reverse split" "round up"` | Variant wording |
| 3 | `"fractional share" "rounded up to the nearest whole share"` | Exact standard phrase |
| 4 | `"fractional shares" "round up" "whole share"` | Variant |
| 5 | `"share consolidation" "rounded up"` | Foreign issuers (6-K, ordinary shares) |
| 6 | `"reverse stock split" "round lot"` | Round-lot protection |
| 7 | `"participant level" "reverse stock split"` | Flag broker-dependent deals |
| 8 | `"registered holder level" OR "record holder level" "reverse"` | Flag broker-dependent deals |
| 9 | `"reverse stock split" "cash in lieu"` | Negatives, logged for completeness and recall checks |
| 10 | `"market effective date" "reverse"` | Catch timing announcements whose treatment is stated elsewhere |

Each query runs with `dateRange=custom&startdt=<today-7d>&enddt=<today>` and pages with `from=`. Results are de-duplicated by accession number.

## Appendix B: Phrase/regex library (seed)

```python
POSITIVE = [
  r"round(?:ed|ing)?\s+up\s+to\s+the\s+(?:nearest|next)\s+whole\s+share",
  r"fraction(?:al)?\s+shares?.{0,250}?round(?:ed)?\s+up",
  r"in\s+lieu\s+of\s+(?:issuing\s+)?(?:any\s+)?fractional\s+shares?.{0,200}?(?:one|a)\s+(?:whole|full)\s+share",
  r"entitled\s+to\s+receive\s+(?:one|a)\s+(?:whole|full)\s+share",
]
ROUND_LOT = [r"round[\s-]lot", r"(?:no\s+fewer|not\s+less)\s+than\s+100\s+shares"]
LEVEL_QUALIFIERS = [r"participant\s+level", r"DTC\s+participant", r"(?:registered|record)\s+holder\s+level", r"holders?\s+of\s+record\s+level"]
NEGATIVE = [r"cash\s+(?:payment\s+)?in\s+lieu", r"paid\s+in\s+cash", r"aggregat\w+.{0,120}?sold", r"round(?:ed)?\s+down",
            r"no\s+fractional\s+shares\s+will\s+be\s+issued.{0,150}?cash"]
MODIFIERS = [r"settled\s+shares", r"acquired\s+(?:after|following)", r"reserves?\s+the\s+right", r"in\s+its\s+(?:sole\s+)?discretion"]
```

## Appendix C: References consulted while drafting
- auto-rsa: https://github.com/NelsonDane/auto-rsa (broker/method table; `docs/BROKERAGES.md` for credential formats)
- itsjafer reverse-stock-split-arbitrage: https://github.com/itsjafer/reverse-stock-split-arbitrage
- SIFMA, *Reverse Stock Splits and Fractional Share Management: A Framework for Market Integrity* (Dec 2025) and issuer best-practices FAQ: https://www.sifma.org/research/white-papers/reverse-stock-splits-and-fractional-share-management-a-framework-for-market-integrity
- Nasdaq reverse-split halt: SEC order on Nasdaq Rules 4120/4753 (2023); SR-NASDAQ-2026-057 (Fed. Reg. Jul 13, 2026)
- Nasdaq reverse split rules/timeline overview (Jun 2026): https://www.securitieslawyer101.com/2026/06/18/nasdaq-reverse-stock-split-rules-timeline/
- Participant-level examples: Beyond Meat 1-for-30 (investors.beyondmeat.com); Auddia; Nexera (eff. Nov 6, 2026)
- Round-lot protection: American Rebel (AREB) Feb 2 and Mar 23, 2026 announcements
- EDGAR FTS endpoint notes: https://orthogonal.info/sec-edgar-full-text-search-api-efts-python/
- Playwright MCP persistent profiles: https://playwright.dev/mcp/configuration/user-profile
- Public Individual API: https://public.com/learn/what-is-trading-api
- Webull OpenAPI: https://www.webull.com/open-api
- Robinhood corporate actions tracker: https://robinhood.com/support/articles/360001301006/corporate-actions-tracker
- Split calendars: dilutiontracker.com/app/reverse-split, stocktitan.net/news/stock-splits.html, tipranks.com/calendars/stock-splits/upcoming
