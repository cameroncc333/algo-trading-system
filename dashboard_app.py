"""
Algorithmic Trading System — Streamlit Dashboard
Built by Cameron Camarotti | github.com/cameroncc333
Run: streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

from config import (
    SECTOR_ETFS, BENCHMARK, INITIAL_CAPITAL, MAX_POSITIONS,
    REBALANCE_FREQUENCY, TRANSACTION_COST_BPS, RSI_PERIOD,
    MOMENTUM_LOOKBACK_DAYS, SHARPE_LOOKBACK_DAYS,
    MOMENTUM_WEIGHT, RSI_WEIGHT, SHARPE_WEIGHT,
    RSI_OVERBOUGHT, RSI_OVERSOLD, STOP_LOSS_PCT,
    MAX_DRAWDOWN_LIMIT, BACKTEST_START, BACKTEST_END, RISK_FREE_RATE
)
from data import fetch_prices
from signals import generate_composite_scores, get_signal_components
from backtest import run_backtest
from walk_forward import run_walk_forward
from journal import load_journal, add_entry, update_entry, get_journal_summary

st.set_page_config(page_title="Algo Trading System", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Strategy Parameters")
st.sidebar.markdown("---")

with st.sidebar.expander("Signal Weights", expanded=True):
    mom_w = st.slider("Momentum Weight", 0.0, 1.0, MOMENTUM_WEIGHT, 0.05)
    rsi_w = st.slider("RSI Weight", 0.0, 1.0, RSI_WEIGHT, 0.05)
    sharpe_w = st.slider("Sharpe Weight", 0.0, 1.0, SHARPE_WEIGHT, 0.05)
    total_w = mom_w + rsi_w + sharpe_w
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_w:.2f} — will be auto-normalized to 1.0")

with st.sidebar.expander("Backtest Settings"):
    max_pos = st.slider("Max Positions", 1, 6, MAX_POSITIONS)
    capital = st.number_input("Initial Capital ($)", 1000, 1000000, INITIAL_CAPITAL, 1000)
    tx_cost = st.slider("Transaction Cost (bps)", 0, 50, TRANSACTION_COST_BPS)
    rebal_freq = st.selectbox("Rebalance Frequency", ["monthly", "weekly"], index=0)

with st.sidebar.expander("Risk Management"):
    max_dd = st.slider("Max Drawdown Limit", 0.05, 0.30, MAX_DRAWDOWN_LIMIT, 0.01)
    stop_loss = st.slider("Stop Loss Pct", 0.03, 0.15, STOP_LOSS_PCT, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Cameron Camarotti**")
st.sidebar.markdown("[GitHub](https://github.com/cameroncc333)")

@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_data():
    return fetch_prices()

@st.cache_data(ttl=3600, show_spinner="Running backtest...")
def cached_backtest(_prices, _capital, _max_pos, _tx_cost, _rebal_freq, _max_dd, _stop_loss, _mom_w, _rsi_w, _sharpe_w):
    return run_backtest(_prices, initial_capital=_capital, max_positions=_max_pos, tx_cost_bps=_tx_cost, rebal_freq=_rebal_freq, max_dd_limit=_max_dd, stop_loss_pct=_stop_loss, mom_weight=_mom_w, rsi_weight=_rsi_w, sharpe_weight=_sharpe_w)

st.title("Algorithmic Trading System")
st.markdown("**Sector Rotation Strategy** — RSI x Momentum x Sharpe signals with systematic backtesting")
st.markdown("---")

try:
    prices = load_data()
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

results = cached_backtest(prices, capital, max_pos, tx_cost, rebal_freq, max_dd, stop_loss, mom_w, rsi_w, sharpe_w)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Signal Analysis", "Backtest Results", "Trade Log", "Paper Trading Journal", "Walk-Forward Validation"])

with tab1:
    st.header("Strategy Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("How It Works")
        st.markdown(f"This system ranks the 11 S&P 500 sector ETFs and goes long the top **{max_pos}** sectors each month.")
        st.markdown(f"**Momentum Weight:** {mom_w:.0%} | **RSI Weight:** {rsi_w:.0%} | **Sharpe Weight:** {sharpe_w:.0%}")
        st.markdown(f"**Stop Loss:** {stop_loss:.0%} | **Circuit Breaker:** {max_dd:.0%} | **TX Cost:** {tx_cost} bps")
    with col2:
        st.subheader("Current Sector Snapshot")
        sector_tickers = list(SECTOR_ETFS.keys())
        latest_prices = prices[sector_tickers].iloc[-1]
        month_ago = prices[sector_tickers].iloc[-21] if len(prices) > 21 else prices[sector_tickers].iloc[0]
        month_return = (latest_prices / month_ago - 1) * 100
        snapshot = pd.DataFrame({"Sector": [SECTOR_ETFS[t] for t in sector_tickers], "Ticker": sector_tickers, "Price": [f"${latest_prices[t]:.2f}" for t in sector_tickers], "1M Return": [f"{month_return[t]:+.1f}%" for t in sector_tickers]}).set_index("Ticker")
        st.dataframe(snapshot, use_container_width=True)
    st.subheader("Sector Performance Heatmap")
    returns_data = {}
    for period_name, days in [("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]:
        if len(prices) > days:
            ret = (prices[sector_tickers].iloc[-1] / prices[sector_tickers].iloc[-days] - 1) * 100
            returns_data[period_name] = ret
    if returns_data:
        heatmap_df = pd.DataFrame(returns_data, index=[SECTOR_ETFS[t] for t in sector_tickers])
        fig_heat = px.imshow(heatmap_df.values, x=heatmap_df.columns.tolist(), y=heatmap_df.index.tolist(), color_continuous_scale="RdYlGn", aspect="auto", text_auto=".1f", labels={"color": "Return %"})
        fig_heat.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    st.header("Signal Analysis")
    signals = get_signal_components(prices)
    composite = signals["composite"]
    sector_tickers = list(SECTOR_ETFS.keys())
    st.subheader("Current Composite Rankings")
    latest_scores = composite.iloc[-1].sort_values(ascending=False)
    latest_scores = latest_scores[latest_scores > -999]
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_rank = go.Figure()
        colors = ["#00d4aa" if i < max_pos else "#4a5568" for i in range(len(latest_scores))]
        fig_rank.add_trace(go.Bar(x=[SECTOR_ETFS.get(t, t) for t in latest_scores.index], y=latest_scores.values, marker_color=colors, text=[f"{v:.2f}" for v in latest_scores.values], textposition="outside"))
        fig_rank.update_layout(title=f"Composite Score — Top {max_pos} Selected (Green)", yaxis_title="Composite Score (z-score)", height=400, showlegend=False)
        st.plotly_chart(fig_rank, use_container_width=True)
    with col2:
        st.markdown("### Selected Sectors")
        for i, (ticker, score) in enumerate(latest_scores.head(max_pos).items()):
            st.markdown(f"**{i+1}. {SECTOR_ETFS[ticker]}** ({ticker}) Score: {score:.3f}")
        excluded = composite.iloc[-1][composite.iloc[-1] <= -999]
        if len(excluded) > 0:
            st.markdown("### Below Trend (Excluded)")
            for ticker in excluded.index:
                st.markdown(f"- {SECTOR_ETFS[ticker]} ({ticker})")
    st.markdown("---")
    st.subheader("Signal Components Over Time")
    selected_sector = st.selectbox("Select Sector", sector_tickers, format_func=lambda x: f"{SECTOR_ETFS[x]} ({x})")
    col1, col2 = st.columns(2)
    with col1:
        rsi_data = signals["rsi"][selected_sector].dropna().tail(252)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data.values, mode="lines", name="RSI", line=dict(color="#6366f1", width=2)))
        fig_rsi.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title=f"RSI ({RSI_PERIOD}-day)", yaxis_title="RSI", height=350, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)
    with col2:
        mom_data = signals["momentum"][selected_sector].dropna().tail(252)
        fig_mom = go.Figure()
        fig_mom.add_trace(go.Scatter(x=mom_data.index, y=mom_data.values * 100, mode="lines", name="Momentum", line=dict(color="#f59e0b", width=2), fill="tozeroy"))
        fig_mom.add_hline(y=0, line_color="gray")
        fig_mom.update_layout(title=f"Momentum ({MOMENTUM_LOOKBACK_DAYS}-day)", yaxis_title="Return %", height=350)
        st.plotly_chart(fig_mom, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        sharpe_data = signals["rolling_sharpe"][selected_sector].dropna().tail(252)
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(x=sharpe_data.index, y=sharpe_data.values, mode="lines", name="Rolling Sharpe", line=dict(color="#10b981", width=2), fill="tozeroy"))
        fig_sharpe.add_hline(y=0, line_color="gray")
        fig_sharpe.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="Good (>1)")
        fig_sharpe.update_layout(title=f"Rolling Sharpe ({SHARPE_LOOKBACK_DAYS}-day)", yaxis_title="Sharpe Ratio", height=350)
        st.plotly_chart(fig_sharpe, use_container_width=True)
    with col4:
        comp_data = composite[selected_sector].dropna().tail(252)
        comp_data = comp_data[comp_data > -999]
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=comp_data.index, y=comp_data.values, mode="lines", name="Composite", line=dict(color="#8b5cf6", width=2), fill="tozeroy"))
        fig_comp.add_hline(y=0, line_color="gray")
        fig_comp.update_layout(title=f"Composite Score — {SECTOR_ETFS[selected_sector]}", yaxis_title="Score (z)", height=350)
        st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    st.header("Backtest Results")
    m = results["metrics"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Return", f"{m['total_return']:+.1%}", f"vs SPY {m['benchmark_return']:+.1%}")
    c2.metric("CAGR", f"{m['cagr']:+.1%}", f"vs SPY {m['benchmark_cagr']:+.1%}")
    c3.metric("Sharpe Ratio", f"{m['sharpe']:.2f}", f"vs SPY {m['benchmark_sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{m['max_drawdown']:.1%}", f"vs SPY {m['benchmark_max_dd']:.1%}")
    c5.metric("Alpha (Annual)", f"{m['alpha']:+.1%}")
    c6.metric("Beta", f"{m['beta']:.2f}")
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=results["portfolio_values"].index, y=results["portfolio_values"].values, name="Strategy", line=dict(color="#6366f1", width=2.5)))
        fig_equity.add_trace(go.Scatter(x=results["benchmark_values"].index, y=results["benchmark_values"].values, name="SPY (Benchmark)", line=dict(color="#9ca3af", width=2, dash="dot")))
        fig_equity.update_layout(title=f"Equity Curve — ${capital:,} Initial Capital", yaxis_title="Portfolio Value ($)", height=450, hovermode="x unified")
        st.plotly_chart(fig_equity, use_container_width=True)
    with col2:
        st.markdown("### Performance Summary")
        summary_data = {"Metric": ["Total Return", "CAGR", "Annual Vol", "Sharpe", "Sortino", "Calmar", "Max Drawdown", "Alpha", "Beta", "Win Rate", "Profit Factor", "Total Trades", "Final Value"], "Strategy": [f"{m['total_return']:+.1%}", f"{m['cagr']:+.1%}", f"{m['annualized_vol']:.1%}", f"{m['sharpe']:.2f}", f"{m['sortino']:.2f}", f"{m['calmar']:.2f}", f"{m['max_drawdown']:.1%}", f"{m['alpha']:+.1%}", f"{m['beta']:.2f}", f"{m['win_rate']:.0%}", f"{m['profit_factor']:.2f}", f"{m['total_trades']}", f"${m['final_value']:,.0f}"], "SPY": [f"{m['benchmark_return']:+.1%}", f"{m['benchmark_cagr']:+.1%}", f"{m['benchmark_vol']:.1%}", f"{m['benchmark_sharpe']:.2f}", "—", "—", f"{m['benchmark_max_dd']:.1%}", "—", "1.00", "—", "—", "—", f"${m['benchmark_final']:,.0f}"]}
        st.dataframe(pd.DataFrame(summary_data).set_index("Metric"), use_container_width=True)
    st.subheader("Drawdown Analysis")
    pv = results["portfolio_values"]
    bv = results["benchmark_values"]
    dd_port = (pv - pv.cummax()) / pv.cummax()
    dd_bench = (bv - bv.cummax()) / bv.cummax()
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_port.index, y=dd_port.values * 100, name="Strategy", fill="tozeroy", line=dict(color="#ef4444", width=1.5)))
    fig_dd.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench.values * 100, name="SPY", fill="tozeroy", line=dict(color="#9ca3af", width=1, dash="dot")))
    fig_dd.add_hline(y=-max_dd * 100, line_dash="dash", line_color="yellow", annotation_text="Circuit Breaker")
    fig_dd.update_layout(title="Underwater Plot (Drawdown %)", yaxis_title="Drawdown %", height=350)
    st.plotly_chart(fig_dd, use_container_width=True)
    st.subheader("Monthly Returns Heatmap")
    port_returns = results["portfolio_values"].pct_change().dropna()
    monthly = port_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({"Year": monthly.index.year, "Month": monthly.index.month, "Return": monthly.values * 100})
    if len(monthly_df) > 0:
        pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Return")
        month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        pivot = pivot.reindex(columns=range(1, 13))
        pivot.columns = [month_labels[c] for c in pivot.columns]
        fig_monthly = px.imshow(pivot.values, x=pivot.columns.tolist(), y=[str(y) for y in pivot.index.tolist()], color_continuous_scale="RdYlGn", aspect="auto", text_auto=".1f", labels={"color": "Return %"})
        fig_monthly.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_monthly, use_container_width=True)

with tab4:
    st.header("Trade Log")
    if not results.get("trades"):
        st.info("No trades executed in backtest.")
    else:
        trades_df = pd.DataFrame(results["trades"])
        trades_df["date"] = pd.to_datetime(trades_df["date"]).dt.strftime("%Y-%m-%d")
        trades_df["sector"] = trades_df["ticker"].map(SECTOR_ETFS)
        trades_df["pnl_pct"] = trades_df["pnl_pct"].apply(lambda x: f"{x:+.2%}")
        trades_df["price"] = trades_df["price"].apply(lambda x: f"${x:.2f}")
        trades_df["shares"] = trades_df["shares"].apply(lambda x: f"{x:.2f}")
        col1, col2, col3 = st.columns(3)
        with col1:
            action_filter = st.multiselect("Action", ["BUY", "SELL"], default=["BUY", "SELL"])
        with col2:
            reason_filter = st.multiselect("Reason", trades_df["reason"].unique().tolist(), default=trades_df["reason"].unique().tolist())
        with col3:
            sector_filter = st.multiselect("Sector", trades_df["sector"].dropna().unique().tolist(), default=trades_df["sector"].dropna().unique().tolist())
        filtered = trades_df[(trades_df["action"].isin(action_filter)) & (trades_df["reason"].isin(reason_filter)) & (trades_df["sector"].isin(sector_filter))]
        st.dataframe(filtered[["date", "sector", "ticker", "action", "reason", "shares", "price", "pnl_pct"]].sort_values("date", ascending=False), use_container_width=True, height=500)
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", len(results["trades"]))
        c2.metric("Buys", len([t for t in results["trades"] if t["action"] == "BUY"]))
        c3.metric("Sells", len([t for t in results["trades"] if t["action"] == "SELL"]))
        c4.metric("Stop Losses Hit", len([t for t in results["trades"] if t["reason"] == "STOP_LOSS"]))

with tab5:
    st.header("Paper Trading Journal")
    st.markdown("Document your monthly decisions: **what the model said -> what you did -> what happened -> what you learned**.")
    journal = get_journal_summary()
    with st.expander("Add New Entry", expanded=True):
        jcol1, jcol2 = st.columns(2)
        with jcol1:
            j_date = st.date_input("Date", value=date.today())
            j_signal = st.text_area("Model Signal", placeholder="What did the composite scores say?")
            j_sectors = st.multiselect("Sectors Recommended", [f"{SECTOR_ETFS[t]} ({t})" for t in SECTOR_ETFS])
            j_confidence = st.slider("Confidence in Signal (1-10)", 1, 10, 5)
        with jcol2:
            j_action = st.text_area("Action Taken", placeholder="What trades did you make?")
            j_rationale = st.text_area("Rationale", placeholder="Why did you follow or deviate?")
            j_context = st.text_area("Market Context", placeholder="VIX level, 10yr yield, Fed stance...")
            j_port_val = st.number_input("Portfolio Value ($)", 0, 1000000, capital)
            j_bench_val = st.number_input("SPY Benchmark Value ($)", 0, 1000000, capital)
        if st.button("Save Entry", type="primary"):
            if j_signal and j_action:
                entry = add_entry(date=j_date.isoformat(), model_signal=j_signal, sectors_recommended=[s.split("(")[1].rstrip(")") for s in j_sectors], action_taken=j_action, rationale=j_rationale, portfolio_value=j_port_val, benchmark_value=j_bench_val, confidence_level=j_confidence, market_context=j_context)
                st.success(f"Entry #{entry['id']} saved!")
                st.rerun()
            else:
                st.warning("Fill in at least Model Signal and Action Taken.")
    st.markdown("---")
    st.subheader("Journal History")
    if journal["total_entries"] == 0:
        st.info("No entries yet. Add your first paper trading decision above.")
    else:
        st.markdown(f"**{journal['total_entries']} entries** | Avg confidence: {journal['avg_confidence']:.1f}/10")
        for entry in reversed(journal["entries"]):
            with st.expander(f"{entry['date']} — Confidence: {entry['confidence_level']}/10"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Model Signal:** {entry['model_signal']}")
                    st.markdown(f"**Sectors:** {', '.join(entry['sectors_recommended'])}")
                    st.markdown(f"**Action:** {entry['action_taken']}")
                with col2:
                    st.markdown(f"**Market Context:** {entry.get('market_context', 'N/A')}")
                    st.markdown(f"**Portfolio:** ${entry['portfolio_value']:,.0f}")
                    if entry.get("outcome"):
                        st.markdown(f"**Outcome:** {entry['outcome']}")
                uc1, uc2 = st.columns(2)
                with uc1:
                    outcome = st.text_input("Outcome", value=entry.get("outcome", ""), key=f"outcome_{entry['id']}")
                with uc2:
                    lessons = st.text_input("Lessons", value=entry.get("lessons_learned", ""), key=f"lessons_{entry['id']}")
                if st.button("Update", key=f"update_{entry['id']}"):
                    update_entry(entry["id"], outcome=outcome, lessons_learned=lessons)
                    st.success("Updated!")
                    st.rerun()

with tab6:
    st.header("Walk-Forward Validation")
    st.markdown("**The overfitting test.** Parameters are optimized on past data only, then tested on unseen future data.")
    st.markdown("---")
    wf_col1, wf_col2 = st.columns(2)
    with wf_col1:
        wf_train = st.slider("Training Window (months)", 12, 36, 24)
    with wf_col2:
        wf_test = st.slider("Test Window (months)", 3, 12, 6)
    if st.button("Run Walk-Forward Validation", type="primary"):
        with st.spinner("Running walk-forward validation (1-3 minutes)..."):
            wf_results = run_walk_forward(prices, train_months=wf_train, test_months=wf_test, initial_capital=capital, tx_cost_bps=tx_cost, rebal_freq=rebal_freq, max_dd_limit=max_dd, stop_loss_pct=stop_loss)
        if "error" in wf_results:
            st.error(wf_results["error"])
        else:
            oos = wf_results["oos_metrics"]
            st.subheader("Out-of-Sample Performance")
            wc1, wc2, wc3, wc4, wc5 = st.columns(5)
            wc1.metric("OOS Total Return", f"{oos.get('total_return', 0):+.1%}")
            wc2.metric("OOS Sharpe", f"{oos.get('sharpe', 0):.2f}", f"vs SPY {oos.get('benchmark_sharpe', 0):.2f}")
            wc3.metric("OOS Max Drawdown", f"{oos.get('max_drawdown', 0):.1%}")
            wc4.metric("Folds Tested", wf_results["n_folds"])
            wc5.metric("Param Combos/Fold", wf_results["n_param_combos"])
            st.markdown("---")
            st.subheader("Out-of-Sample Equity Curve")
            oos_eq = wf_results["oos_equity"]
            bench_eq = wf_results["benchmark_equity"]
            if len(oos_eq) > 0:
                fig_wf = go.Figure()
                fig_wf.add_trace(go.Scatter(x=oos_eq.index, y=oos_eq.values, name="Strategy (OOS)", line=dict(color="#6366f1", width=2.5)))
                if len(bench_eq) > 0:
                    fig_wf.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values, name="SPY (Benchmark)", line=dict(color="#9ca3af", width=2, dash="dot")))
                for fold in wf_results["folds"]:
                    fig_wf.add_vline(x=fold["test_start"], line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
                fig_wf.update_layout(title="Out-of-Sample Equity", yaxis_title="Portfolio Value ($)", height=450, hovermode="x unified")
                st.plotly_chart(fig_wf, use_container_width=True)
            st.subheader("Fold-by-Fold Results")
            fold_df = pd.DataFrame(wf_results["folds"])
            fold_display = fold_df[["fold", "train_start", "train_end", "test_start", "test_end", "train_sharpe", "test_sharpe", "test_return", "test_max_dd", "test_alpha"]].copy()
            fold_display["train_start"] = pd.to_datetime(fold_display["train_start"]).dt.strftime("%Y-%m-%d")
            fold_display["train_end"] = pd.to_datetime(fold_display["train_end"]).dt.strftime("%Y-%m-%d")
            fold_display["test_start"] = pd.to_datetime(fold_display["test_start"]).dt.strftime("%Y-%m-%d")
            fold_display["test_end"] = pd.to_datetime(fold_display["test_end"]).dt.strftime("%Y-%m-%d")
            fold_display["train_sharpe"] = fold_display["train_sharpe"].apply(lambda x: f"{x:.2f}")
            fold_display["test_sharpe"] = fold_display["test_sharpe"].apply(lambda x: f"{x:.2f}")
            fold_display["test_return"] = fold_display["test_return"].apply(lambda x: f"{x:+.1%}")
            fold_display["test_max_dd"] = fold_display["test_max_dd"].apply(lambda x: f"{x:.1%}")
            fold_display["test_alpha"] = fold_display["test_alpha"].apply(lambda x: f"{x:+.1%}")
            fold_display.columns = ["Fold", "Train Start", "Train End", "Test Start", "Test End", "Train Sharpe", "Test Sharpe", "Test Return", "Test Max DD", "Test Alpha"]
            st.dataframe(fold_display, use_container_width=True, hide_index=True)
            st.subheader("Selected Parameters by Fold")
            params_per_fold = pd.DataFrame([{"Fold": f["fold"], **f["best_params"]} for f in wf_results["folds"]])
            st.dataframe(params_per_fold, use_container_width=True, hide_index=True)
            st.subheader("Parameter Stability")
            st.markdown("If the same value is selected across most folds, the signal is robust.")
            stab = wf_results["param_stability"]
            for param_name, info in stab.items():
                stability_pct = info["stability"] * 100
                indicator = "STRONG" if stability_pct >= 60 else "MODERATE" if stability_pct >= 40 else "WEAK"
                st.markdown(f"**{param_name}**: Most common = {info['most_common']} (selected in {stability_pct:.0f}% of folds) — Stability: {indicator} — Distribution: {info['values']}")
    else:
        st.info("Click the button above to run walk-forward validation. Tests ~81 parameter combinations across multiple time folds.")

st.markdown("---")
st.markdown("**Algo Trading System v1.0** | Built by Cameron Camarotti | Not financial advice")
