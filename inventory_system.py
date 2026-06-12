"""
Inventory Management System
Naturub Exports International (Pvt) Ltd
Built on GRN, Issued, and Requested datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io, os

st.set_page_config(page_title="Inventory Management System", page_icon="📦", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding-top: 1.5rem; }
.kpi { background:#0f172a; border:1px solid #1e3a5f; border-radius:10px; padding:1.2rem 1rem; text-align:center; }
.kpi .label { color:#64748b; font-size:0.7rem; text-transform:uppercase; letter-spacing:2px; }
.kpi .value { color:#38bdf8; font-size:1.8rem; font-weight:700; font-family:'IBM Plex Mono'; }
.kpi .sub   { color:#94a3b8; font-size:0.75rem; }
.alert-red   { background:#1c0a0a; border-left:4px solid #ef4444; border-radius:0 8px 8px 0; padding:10px 14px; color:#fca5a5; font-size:13px; margin:4px 0; }
.alert-green { background:#052e16; border-left:4px solid #22c55e; border-radius:0 8px 8px 0; padding:10px 14px; color:#86efac; font-size:13px; margin:4px 0; }
.alert-amber { background:#1c1500; border-left:4px solid #f59e0b; border-radius:0 8px 8px 0; padding:10px 14px; color:#fcd34d; font-size:13px; margin:4px 0; }
.alert-blue  { background:#0c1445; border-left:4px solid #38bdf8; border-radius:0 8px 8px 0; padding:10px 14px; color:#bae6fd; font-size:13px; margin:4px 0; }
.section-title { color:#38bdf8; font-size:1rem; font-weight:600; text-transform:uppercase; letter-spacing:2px; margin:1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Login ─────────────────────────────────────────────────────────────────────
def load_users():
    users_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.txt")
    if not os.path.exists(users_path):
        users_path = "users.txt"
    if not os.path.exists(users_path):
        return {"admin": "admin123"}
    users = {}
    with open(users_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                users[parts[0].strip()] = parts[1].strip()
    return users

def show_login():
    st.markdown("""
    <div style='text-align:center; padding-top:3rem;'>
        <span style='font-size:3rem;'>📦</span>
        <h1 style='color:#38bdf8; font-family:IBM Plex Mono; margin:0.5rem 0;'>Inventory Management System</h1>
        <p style='color:#64748b;'>Naturub Exports International (Pvt) Ltd</p>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔐 Sign In", use_container_width=True, type="primary"):
            users = load_users()
            if username in users and users[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        st.markdown('<p style="text-align:center;color:#475569;font-size:0.75rem;margin-top:1rem;">Contact your administrator for access.</p>', unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    grn    = pd.read_excel("/mnt/user-data/uploads/GRN_Dataset.xlsx",      parse_dates=["TRANSACTION_DATE"])
    issued = pd.read_excel("/mnt/user-data/uploads/Issued_Dataset.xlsx",   parse_dates=["TRANSACTION_DATE"])
    req    = pd.read_excel("/mnt/user-data/uploads/Requested_Data.xlsx",   parse_dates=["REQUESTED_ON"])
    # clean
    for df in [grn, issued]:
        df["ITEM_CODE"]        = df["ITEM_CODE"].astype(str).str.strip()
        df["ITEM_DESCRIPTION"] = df["ITEM_DESCRIPTION"].astype(str).str.strip()
        df["QTY"]              = pd.to_numeric(df["QTY"], errors="coerce").fillna(0)
        df["BALANCE_QTY"]      = pd.to_numeric(df["BALANCE_QTY"], errors="coerce")
        df["REORDER_QTY"]      = pd.to_numeric(df["REORDER_QTY"], errors="coerce").fillna(0)
        df["SIH"]              = pd.to_numeric(df["SIH"], errors="coerce").fillna(0)
    grn["Lead Time "] = pd.to_numeric(grn["Lead Time "], errors="coerce").fillna(45)
    return grn, issued, req

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_item_info(grn, issued, item_code):
    g = grn[grn["ITEM_CODE"] == item_code]
    i = issued[issued["ITEM_CODE"] == item_code]
    if g.empty and i.empty:
        return None
    desc        = g["ITEM_DESCRIPTION"].iloc[0] if not g.empty else i["ITEM_DESCRIPTION"].iloc[0]
    unit        = g["MEASUER_UNIT"].iloc[0]      if not g.empty else i["MEASUER_UNIT"].iloc[0]
    reorder_qty = g["REORDER_QTY"].iloc[0]       if not g.empty else i["REORDER_QTY"].iloc[0]
    sih         = g["SIH"].iloc[-1]              if not g.empty else i["SIH"].iloc[-1]
    lead_time   = int(g["Lead Time "].iloc[0])   if not g.empty and g["Lead Time "].notna().any() else 45
    stock_type  = g["STOCK_TYPE"].iloc[0]        if not g.empty else i["STOCK_TYPE"].iloc[0]
    return {"desc": desc, "unit": unit, "reorder_qty": reorder_qty,
            "sih": sih, "lead_time": lead_time, "stock_type": stock_type,
            "grn": g, "issued": i}

def monthly_movement(grn_item, issued_item):
    grn_m = (grn_item.set_index("TRANSACTION_DATE")["QTY"]
             .resample("ME").sum().reset_index()
             .rename(columns={"TRANSACTION_DATE":"month","QTY":"received"}))
    iss_m = (issued_item.set_index("TRANSACTION_DATE")["QTY"]
             .resample("ME").sum().reset_index()
             .rename(columns={"TRANSACTION_DATE":"month","QTY":"issued"}))
    m = pd.merge(grn_m, iss_m, on="month", how="outer").fillna(0)
    m = m.sort_values("month")
    return m

def calc_reorder_recommendation(issued_item, lead_time):
    if issued_item.empty: return 0, 0
    daily = issued_item.set_index("TRANSACTION_DATE")["QTY"].resample("D").sum()
    avg_daily = daily.mean()
    safety    = avg_daily * 7   # 1 week safety stock
    rec_reorder = round((avg_daily * lead_time) + safety, 2)
    return round(avg_daily, 2), rec_reorder

# ── Main App ──────────────────────────────────────────────────────────────────
def show_app():
    # Sidebar
    with st.sidebar:
        st.markdown("## 📦 Inventory System")
        st.markdown("**Naturub Exports**")
        st.markdown("---")
        st.markdown(f"👤 **{st.session_state.get('username','')}**")
        page = st.radio("Navigation", [
            "🔍 Item Lookup",
            "📊 Stock Movement",
            "⚠️ Reorder Alerts",
            "🛒 Order Evaluator",
            "📅 Order Date Planner",
            "📈 Reorder Optimizer",
        ])
        st.markdown("---")
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    with st.spinner("Loading data..."):
        grn, issued, req = load_data()

    item_list = sorted(set(grn["ITEM_CODE"].unique()) | set(issued["ITEM_CODE"].unique()))

    # ── PAGE 1: Item Lookup ──────────────────────────────────────────────────
    if page == "🔍 Item Lookup":
        st.markdown("# 🔍 Item Lookup")
        st.markdown("Search by item code or job number to view current stock details.")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            search_code = st.text_input("Enter Item Code", placeholder="e.g. YPOL11267").strip().upper()
        with col2:
            search_job = st.text_input("Enter Job Number", placeholder="e.g. 22-Y-27238-D005").strip()

        # job number search
        if search_job:
            grn_job    = grn[grn["JOB_NO"].astype(str).str.contains(search_job, na=False)]
            issued_job = issued[issued["JOB_NO"].astype(str).str.contains(search_job, na=False)]
            if grn_job.empty and issued_job.empty:
                st.warning(f"No records found for job number: {search_job}")
            else:
                items_in_job = set(grn_job["ITEM_CODE"].tolist()) | set(issued_job["ITEM_CODE"].tolist())
                st.success(f"Found {len(items_in_job)} item(s) for job {search_job}")
                search_code = st.selectbox("Select item from job", sorted(items_in_job))

        if search_code:
            info = get_item_info(grn, issued, search_code)
            if not info:
                st.error(f"Item code **{search_code}** not found in the system.")
            else:
                st.markdown(f"### {search_code}")
                st.markdown(f"**{info['desc']}**")
                st.markdown(f"*{info['stock_type']} | Unit: {info['unit']}*")
                st.markdown("")

                c1,c2,c3,c4 = st.columns(4)
                status_color = "#22c55e" if info['sih'] > info['reorder_qty'] else "#ef4444"
                for col, val, label, sub in [
                    (c1, f"{info['sih']:,.1f}", "Stock in Hand", info['unit']),
                    (c2, f"{info['reorder_qty']:,.1f}", "Reorder Point", info['unit']),
                    (c3, f"{info['lead_time']} days", "Lead Time", "supplier delivery"),
                    (c4, len(info['grn']), "GRN Transactions", "total receipts"),
                ]:
                    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

                st.markdown("")
                if info['sih'] <= info['reorder_qty']:
                    st.markdown(f'<div class="alert-red">⚠️ BELOW REORDER POINT — Current stock ({info["sih"]:,.1f} {info["unit"]}) is at or below reorder level ({info["reorder_qty"]:,.1f} {info["unit"]}). Immediate action required!</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-green">✅ STOCK OK — Current stock is above reorder point.</div>', unsafe_allow_html=True)

                # Recent transactions
                st.markdown('<p class="section-title">Recent Transactions</p>', unsafe_allow_html=True)
                recent_grn = info['grn'].sort_values("TRANSACTION_DATE", ascending=False).head(10)[
                    ["TRANSACTION_DATE","QTY","GRN_NO","JOB_NO","COMPANY"]].rename(
                    columns={"TRANSACTION_DATE":"Date","QTY":"Qty Received","GRN_NO":"GRN No","JOB_NO":"Job No","COMPANY":"Supplier"})
                recent_iss = info['issued'].sort_values("TRANSACTION_DATE", ascending=False).head(10)[
                    ["TRANSACTION_DATE","QTY","JOB_NO","ISSUE_BY"]].rename(
                    columns={"TRANSACTION_DATE":"Date","QTY":"Qty Issued","JOB_NO":"Job No","ISSUE_BY":"Issued By"})

                t1, t2 = st.tabs(["📥 Recent Receipts (GRN)", "📤 Recent Issues"])
                with t1: st.dataframe(recent_grn, use_container_width=True, hide_index=True)
                with t2: st.dataframe(recent_iss, use_container_width=True, hide_index=True)

    # ── PAGE 2: Stock Movement ────────────────────────────────────────────────
    elif page == "📊 Stock Movement":
        st.markdown("# 📊 Stock Movement Trends")
        st.markdown("Interactive graphs showing how inventory has changed over time.")
        st.markdown("---")

        item_code = st.selectbox("Select Item", item_list, key="mv_item")
        info = get_item_info(grn, issued, item_code)

        if info:
            st.markdown(f"**{info['desc']}** | Unit: {info['unit']}")
            m = monthly_movement(info['grn'], info['issued'])

            if not m.empty:
                # Movement chart
                fig = go.Figure()
                fig.add_trace(go.Bar(x=m["month"], y=m["received"], name="Received (GRN)",
                                     marker_color="#38bdf8", opacity=0.8))
                fig.add_trace(go.Bar(x=m["month"], y=m["issued"], name="Issued",
                                     marker_color="#f97316", opacity=0.8))
                if info['reorder_qty'] > 0:
                    fig.add_hline(y=info['reorder_qty'], line_dash="dash",
                                  line_color="#ef4444", annotation_text="Reorder Point")
                fig.update_layout(barmode="group", template="plotly_dark",
                                  title="Monthly Stock Movement",
                                  xaxis_title="Month", yaxis_title=f"Quantity ({info['unit']})",
                                  height=400, legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)

                # Net stock trend
                m["net"] = m["received"] - m["issued"]
                m["cumulative"] = m["net"].cumsum()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=m["month"], y=m["cumulative"],
                                          mode="lines+markers", name="Cumulative Net Stock",
                                          line=dict(color="#22c55e", width=2)))
                if info['reorder_qty'] > 0:
                    # Highlight months below reorder
                    below = m[m["cumulative"] < info['reorder_qty']]
                    if not below.empty:
                        fig2.add_trace(go.Scatter(x=below["month"], y=below["cumulative"],
                                                   mode="markers", name="Below Reorder Point",
                                                   marker=dict(color="#ef4444", size=10, symbol="x")))
                fig2.update_layout(template="plotly_dark", title="Cumulative Stock Trend",
                                   xaxis_title="Month", yaxis_title=f"Quantity ({info['unit']})",
                                   height=350)
                st.plotly_chart(fig2, use_container_width=True)

                # Months below reorder
                if info['reorder_qty'] > 0:
                    below_months = m[m["cumulative"] < info['reorder_qty']]
                    if not below_months.empty:
                        st.markdown('<p class="section-title">⚠️ Months Below Reorder Point</p>', unsafe_allow_html=True)
                        for _, row in below_months.iterrows():
                            st.markdown(f'<div class="alert-red">📅 {row["month"].strftime("%B %Y")} — Stock: {row["cumulative"]:,.1f} {info["unit"]} (Reorder point: {info["reorder_qty"]:,.1f})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-green">✅ Stock never dropped below reorder point in this period.</div>', unsafe_allow_html=True)

    # ── PAGE 3: Reorder Alerts ────────────────────────────────────────────────
    elif page == "⚠️ Reorder Alerts":
        st.markdown("# ⚠️ Reorder Alerts")
        st.markdown("Items currently at or below their reorder point.")
        st.markdown("---")

        with st.spinner("Scanning all items..."):
            # Get latest SIH and reorder for all items
            latest_grn = (grn.sort_values("TRANSACTION_DATE")
                          .groupby("ITEM_CODE").last()
                          .reset_index()[["ITEM_CODE","ITEM_DESCRIPTION","SIH","REORDER_QTY","STOCK_TYPE","MEASUER_UNIT","Lead Time "]])
            latest_grn.columns = ["ITEM_CODE","ITEM_DESCRIPTION","SIH","REORDER_QTY","STOCK_TYPE","UNIT","LEAD_TIME"]
            latest_grn["SIH"]        = pd.to_numeric(latest_grn["SIH"], errors="coerce").fillna(0)
            latest_grn["REORDER_QTY"]= pd.to_numeric(latest_grn["REORDER_QTY"], errors="coerce").fillna(0)
            alerts = latest_grn[(latest_grn["REORDER_QTY"] > 0) &
                                 (latest_grn["SIH"] <= latest_grn["REORDER_QTY"])].copy()
            alerts["Stock Gap"] = alerts["REORDER_QTY"] - alerts["SIH"]
            alerts = alerts.sort_values("Stock Gap", ascending=False)

        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="kpi"><div class="label">Items Below Reorder</div><div class="value">{len(alerts)}</div><div class="sub">require attention</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="kpi"><div class="label">Total Items Tracked</div><div class="value">{len(latest_grn)}</div><div class="sub">in system</div></div>', unsafe_allow_html=True)
        pct = round(len(alerts)/max(len(latest_grn),1)*100, 1)
        col3.markdown(f'<div class="kpi"><div class="label">Alert Rate</div><div class="value">{pct}%</div><div class="sub">of items at risk</div></div>', unsafe_allow_html=True)

        st.markdown("")
        stock_type_filter = st.selectbox("Filter by Stock Type", ["All"] + sorted(latest_grn["STOCK_TYPE"].dropna().unique().tolist()))
        disp = alerts if stock_type_filter == "All" else alerts[alerts["STOCK_TYPE"] == stock_type_filter]

        st.dataframe(disp[["ITEM_CODE","ITEM_DESCRIPTION","STOCK_TYPE","UNIT","SIH","REORDER_QTY","Stock Gap","LEAD_TIME"]]
                     .rename(columns={"ITEM_CODE":"Item Code","ITEM_DESCRIPTION":"Description",
                                      "STOCK_TYPE":"Type","UNIT":"Unit","SIH":"Stock in Hand",
                                      "REORDER_QTY":"Reorder Point","Stock Gap":"Gap","LEAD_TIME":"Lead Time (days)"})
                     .round(2), use_container_width=True, hide_index=True)

        csv = disp.to_csv(index=False).encode()
        st.download_button("⬇️ Download Alert List", csv, "reorder_alerts.csv", "text/csv")

    # ── PAGE 4: Order Evaluator ───────────────────────────────────────────────
    elif page == "🛒 Order Evaluator":
        st.markdown("# 🛒 Order Evaluator")
        st.markdown("Check if current stock is sufficient to fulfill an incoming order.")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            item_code  = st.selectbox("Select Item", item_list, key="oe_item")
            order_qty  = st.number_input("Order Requires (Qty)", min_value=0.0, step=10.0, value=1000.0)
        with col2:
            order_unit = st.text_input("Unit", value="Kg")
            order_date = st.date_input("Required By Date", value=datetime.today() + timedelta(days=30))

        if st.button("🔍 Evaluate Order", type="primary"):
            info = get_item_info(grn, issued, item_code)
            if info:
                available = info['sih']
                shortfall  = max(0, order_qty - available)
                sufficient = available >= order_qty

                st.markdown("### Evaluation Result")
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="kpi"><div class="label">Available Stock</div><div class="value">{available:,.1f}</div><div class="sub">{info["unit"]}</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="kpi"><div class="label">Order Required</div><div class="value">{order_qty:,.1f}</div><div class="sub">{info["unit"]}</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="kpi"><div class="label">Shortfall</div><div class="value">{shortfall:,.1f}</div><div class="sub">{info["unit"]}</div></div>', unsafe_allow_html=True)

                st.markdown("")
                if sufficient:
                    st.markdown(f'<div class="alert-green">✅ SUFFICIENT STOCK — Available stock ({available:,.1f} {info["unit"]}) covers the order requirement ({order_qty:,.1f} {info["unit"]}). Order can proceed.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-red">❌ INSUFFICIENT STOCK — Shortfall of {shortfall:,.1f} {info["unit"]}. A purchase order must be placed immediately.</div>', unsafe_allow_html=True)
                    days_to_required = (pd.Timestamp(order_date) - pd.Timestamp.today()).days
                    if days_to_required < info['lead_time']:
                        st.markdown(f'<div class="alert-amber">⚠️ URGENT — Only {days_to_required} days until required date but lead time is {info["lead_time"]} days. Expedited ordering needed!</div>', unsafe_allow_html=True)
                    else:
                        order_by = pd.Timestamp(order_date) - timedelta(days=info['lead_time'])
                        st.markdown(f'<div class="alert-blue">📅 Place order by: <strong>{order_by.strftime("%d %B %Y")}</strong> to ensure delivery before {pd.Timestamp(order_date).strftime("%d %B %Y")}</div>', unsafe_allow_html=True)

    # ── PAGE 5: Order Date Planner ────────────────────────────────────────────
    elif page == "📅 Order Date Planner":
        st.markdown("# 📅 Order Date Planner")
        st.markdown("Calculate the latest date an order should be placed to ensure on-time delivery.")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            item_code    = st.selectbox("Select Item / Material", item_list, key="odp_item")
            required_date = st.date_input("Material Required By", value=datetime.today() + timedelta(days=60))
        with col2:
            custom_lead  = st.number_input("Override Lead Time (days, 0 = use system value)", min_value=0, value=0)

        if st.button("📅 Calculate Order Date", type="primary"):
            info = get_item_info(grn, issued, item_code)
            if info:
                lead = custom_lead if custom_lead > 0 else info['lead_time']
                order_by = pd.Timestamp(required_date) - timedelta(days=lead)
                today    = pd.Timestamp.today()
                days_left = (order_by - today).days

                st.markdown("### Order Planning Result")
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="kpi"><div class="label">Required By</div><div class="value">{pd.Timestamp(required_date).strftime("%d %b")}</div><div class="sub">{pd.Timestamp(required_date).strftime("%Y")}</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="kpi"><div class="label">Lead Time</div><div class="value">{lead} days</div><div class="sub">supplier delivery</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="kpi"><div class="label">Order By</div><div class="value">{order_by.strftime("%d %b")}</div><div class="sub">{order_by.strftime("%Y")}</div></div>', unsafe_allow_html=True)

                st.markdown("")
                if days_left < 0:
                    st.markdown(f'<div class="alert-red">🚨 OVERDUE — Order should have been placed {abs(days_left)} days ago! Expedite immediately.</div>', unsafe_allow_html=True)
                elif days_left == 0:
                    st.markdown(f'<div class="alert-amber">⚠️ ORDER TODAY — Last possible day to place the order for {info["desc"][:60]}.</div>', unsafe_allow_html=True)
                elif days_left <= 7:
                    st.markdown(f'<div class="alert-amber">⚠️ URGENT — Only {days_left} days left to place order. Recommended order date: <strong>{order_by.strftime("%d %B %Y")}</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-green">✅ Recommended order date: <strong>{order_by.strftime("%d %B %Y")}</strong> — {days_left} days from today.</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="alert-blue" style="margin-top:10px;">
                📦 <strong>{info['desc'][:80]}</strong><br>
                Required by: {pd.Timestamp(required_date).strftime('%d %B %Y')} &nbsp;→&nbsp;
                Recommended order date: <strong>{order_by.strftime('%d %B %Y')}</strong>
                </div>""", unsafe_allow_html=True)

    # ── PAGE 6: Reorder Optimizer ─────────────────────────────────────────────
    elif page == "📈 Reorder Optimizer":
        st.markdown("# 📈 Reorder Level Optimizer")
        st.markdown("Compare system-recommended reorder levels against current company-defined levels.")
        st.markdown("---")

        item_code = st.selectbox("Select Item", item_list, key="ro_item")
        info = get_item_info(grn, issued, item_code)

        if info and not info['issued'].empty:
            avg_daily, rec_reorder = calc_reorder_recommendation(info['issued'], info['lead_time'])
            current_reorder = info['reorder_qty']

            st.markdown(f"**{info['desc']}** | Lead Time: {info['lead_time']} days")
            st.markdown("")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="kpi"><div class="label">Avg Daily Usage</div><div class="value">{avg_daily:,.1f}</div><div class="sub">{info["unit"]} / day</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="kpi"><div class="label">System Recommendation</div><div class="value">{rec_reorder:,.1f}</div><div class="sub">{info["unit"]}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="kpi"><div class="label">Current Reorder Point</div><div class="value">{current_reorder:,.1f}</div><div class="sub">{info["unit"]}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            diff = rec_reorder - current_reorder
            if abs(diff) < 0.01:
                st.markdown('<div class="alert-green">✅ Current reorder level matches the system recommendation perfectly.</div>', unsafe_allow_html=True)
            elif diff > 0:
                st.markdown(f'<div class="alert-amber">⚠️ UNDERSTOCKED RISK — System recommends reorder point of {rec_reorder:,.1f} {info["unit"]}, which is {diff:,.1f} higher than current setting ({current_reorder:,.1f}). Consider increasing the reorder level.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-blue">ℹ️ CONSERVATIVE — Current reorder point ({current_reorder:,.1f}) is {abs(diff):,.1f} higher than the system recommendation ({rec_reorder:,.1f}). You have extra buffer built in.</div>', unsafe_allow_html=True)

            # Comparison chart
            fig = go.Figure(go.Bar(
                x=["Current Reorder Point", "System Recommendation", "Stock in Hand"],
                y=[current_reorder, rec_reorder, info['sih']],
                marker_color=["#f97316","#38bdf8","#22c55e"],
                text=[f"{current_reorder:,.1f}", f"{rec_reorder:,.1f}", f"{info['sih']:,.1f}"],
                textposition="outside"
            ))
            fig.update_layout(template="plotly_dark",
                              title="Reorder Level Comparison",
                              yaxis_title=f"Quantity ({info['unit']})", height=380)
            st.plotly_chart(fig, use_container_width=True)

            # Monthly usage trend
            monthly_usage = (info['issued'].set_index("TRANSACTION_DATE")["QTY"]
                             .resample("ME").sum().reset_index())
            fig2 = px.line(monthly_usage, x="TRANSACTION_DATE", y="QTY",
                           title="Monthly Consumption Trend",
                           labels={"TRANSACTION_DATE":"Month","QTY":f"Qty Used ({info['unit']})"},
                           template="plotly_dark")
            fig2.add_hline(y=rec_reorder, line_dash="dash", line_color="#38bdf8",
                           annotation_text="Recommended Reorder")
            if current_reorder > 0:
                fig2.add_hline(y=current_reorder, line_dash="dot", line_color="#f97316",
                               annotation_text="Current Reorder")
            st.plotly_chart(fig2, use_container_width=True)

# ── Entry ─────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    show_login()
else:
    show_app()
