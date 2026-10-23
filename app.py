import hashlib
import io

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from quality_checks import run_quality_checks
from validation_rules import validate_dataset
from anomaly_detection import detect_anomalies
from database import (
    init_db,
    save_scan,
    save_scan_issues,
    get_scan_history,
    get_scan_issues,
)


# ============================================================
# PAGE CONFIGURATION AND VISUAL THEME
# ============================================================

st.set_page_config(
    page_title="DataSentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
      --bg: #080b12; --surface: #101622; --surface2: #151d2b;
      --line: #263247; --text: #f3f6ff; --muted: #98a7bd;
      --purple: #8b7cff; --cyan: #52d6e8; --green: #45d6a0;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
      color: var(--text);
      background:
        radial-gradient(ellipse at 72% -8%, rgba(103, 84, 255, .16), transparent 34%),
        radial-gradient(ellipse at 2% 22%, rgba(52, 199, 221, .07), transparent 25%),
        #080b12;
    }
    [data-testid="stHeader"] { background: rgba(8,11,18,.96); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1480px; padding: 1.25rem 2.3rem 2.2rem; }
    h1,h2,h3,h4 { color: #f3f6ff !important; font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
    h1 { font-size: 2.05rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.5rem !important; } h3 { font-size: 1.12rem !important; }
    p,li,label,.stMarkdown { color: #c5cee0; }
    [data-testid="stCaptionContainer"] { color: #8e9cb4; }
    a { color: #b3aaff !important; }

    section[data-testid="stSidebar"] {
      background: linear-gradient(180deg,#0e1421 0%,#0a0f19 100%);
      border-right: 1px solid #202b3d;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.15rem; }
    section[data-testid="stSidebar"] * { color: #e9eeff; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] .stCaption { color: #8998b2; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
      padding: .58rem .72rem; border-radius: 10px; margin: 2px 0;
      border: 1px solid transparent; transition: .18s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#182236; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
      background: linear-gradient(100deg,rgba(126,106,255,.20),rgba(61,199,220,.07));
      border-color: rgba(139,124,255,.35);
    }

    .site-header {
      display:flex; align-items:center; justify-content:space-between; gap:20px;
      padding:15px 20px; margin:0 0 25px; border:1px solid #263147;
      border-radius:16px; background:rgba(15,21,33,.92);
      box-shadow:0 12px 32px rgba(0,0,0,.17);
    }
    .brand-lockup { display:flex; align-items:center; gap:12px; }
    .brand-mark {
      display:flex; align-items:center; justify-content:center; width:42px; height:42px;
      border-radius:12px; color:#fff; font-size:20px; font-weight:800;
      background:linear-gradient(135deg,#7b68ff,#39bcd3);
      box-shadow:0 7px 22px rgba(113,96,245,.25);
    }
    .brand-name { color:#f7f8ff; font:700 17px 'Space Grotesk',sans-serif; letter-spacing:-.4px; }
    .brand-sub { color:#8594ae; font-size:10px; letter-spacing:1.8px; font-weight:700; margin-top:2px; }
    .header-right { display:flex; align-items:center; gap:10px; color:#9aa8bf; font-size:12px; }
    .status-pill { display:inline-flex; align-items:center; gap:7px; padding:7px 10px; border:1px solid #25483f; border-radius:999px; background:#10231f; color:#76e4bb; font-weight:700; }
    .status-dot { width:7px; height:7px; background:#45d6a0; border-radius:50%; box-shadow:0 0 10px rgba(69,214,160,.65); }
    .page-intro { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin:4px 0 20px; }
    .eyebrow { color:#a79bff; font-size:10px; letter-spacing:2px; font-weight:800; text-transform:uppercase; margin-bottom:7px; }
    .intro-copy { color:#98a7bd; font-size:13px; line-height:1.65; max-width:700px; margin-top:6px; }
    .hero {
      position:relative; overflow:hidden; border:1px solid #393c6d; border-radius:22px;
      padding:34px 36px; margin-bottom:25px;
      background:linear-gradient(115deg,#141a2d 0%,#20264b 58%,#28275d 100%);
      box-shadow:0 18px 46px rgba(0,0,0,.22);
    }
    .hero:after { content:''; position:absolute; width:250px; height:250px; right:-65px; top:-105px; border-radius:50%; border:1px solid rgba(175,166,255,.25); box-shadow:0 0 0 25px rgba(175,166,255,.035),0 0 0 52px rgba(175,166,255,.025); }
    .hero-eyebrow { color:#b4abff; font-size:10px; font-weight:800; letter-spacing:2.3px; margin-bottom:12px; }
    .hero-title { color:#fff; font:700 clamp(29px,3vw,43px)/1.12 'Space Grotesk',sans-serif; letter-spacing:-1.4px; }
    .hero-copy { color:#c0cbe2; font-size:13px; line-height:1.8; margin-top:13px; max-width:650px; }
    .panel {
      border:1px solid #253147; border-radius:16px; padding:19px 20px;
      background:linear-gradient(145deg,rgba(18,25,39,.98),rgba(13,18,29,.98));
      box-shadow:0 9px 24px rgba(0,0,0,.12); margin-bottom:12px;
    }
    .panel-title { color:#f0f3ff; font:650 14px 'Space Grotesk',sans-serif; margin-bottom:4px; }
    .panel-sub { color:#8999b3; font-size:11px; line-height:1.6; }
    .feature-card { background:linear-gradient(145deg,#151d2d,#101622); border:1px solid #29364b; border-radius:15px; padding:20px; min-height:145px; }
    .feature-title { color:#f0f3ff; font-weight:700; font-size:15px; margin-bottom:9px; }
    .feature-copy { color:#9eabc2; font-size:12px; line-height:1.75; }
    .section-kicker { color:#9b8eff; font-size:10px; font-weight:800; letter-spacing:1.8px; text-transform:uppercase; margin-bottom:7px; }
    div[data-testid="stMetric"] {
      position:relative; overflow:hidden; min-height:122px; padding:19px 20px;
      background:linear-gradient(145deg,#151e2e,#101622); border:1px solid #2a354b;
      border-radius:15px; box-shadow:0 9px 25px rgba(0,0,0,.14);
      transition:transform .18s ease,border-color .18s ease;
    }
    div[data-testid="stMetric"]:before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#8b7cff,#52d6e8); }
    div[data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:#555184; }
    div[data-testid="stMetricLabel"] { color:#9caac1 !important; font-size:.8rem; font-weight:600; }
    div[data-testid="stMetricValue"] { color:#f5f7ff !important; font:700 1.85rem 'Space Grotesk',sans-serif; }
    div[data-testid="stMetricDelta"] { color:#45d6a0 !important; }
    .stButton > button,.stDownloadButton > button {
      min-height:42px; border-radius:10px; font-weight:700; border:1px solid #303c53;
      background:#182235; color:#edf1ff; transition:all .18s ease;
    }
    .stButton > button:hover,.stDownloadButton > button:hover { border-color:#8b7cff; color:#fff; background:#222c43; }
    .stButton > button[kind="primary"] { background:linear-gradient(105deg,#7967ff,#6557df 65%,#278da9); color:#fff; border:1px solid #7970da; box-shadow:0 7px 22px rgba(112,94,247,.23); }
    .stTextInput input,.stNumberInput input,.stTextArea textarea,[data-baseweb="select"] > div,[data-testid="stDateInput"] input {
      border-radius:9px !important; border-color:#303c53 !important; background:#111927 !important; color:#edf1ff !important;
    }
    [data-baseweb="popover"] { background:#141c2b !important; }
    [data-testid="stFileUploader"] { background:linear-gradient(145deg,#151e2e,#101622); border:1px dashed #655cc1; border-radius:15px; padding:13px; }
    [data-testid="stFileUploader"] section { background:transparent; }
    [data-testid="stFileUploader"] button { background:#252e44; color:#fff; border:1px solid #414c68; }
    [data-testid="stDataFrame"],[data-testid="stTable"] { background:#101622; border:1px solid #29364b; border-radius:12px; overflow:hidden; }
    .stTabs [data-baseweb="tab-list"] { gap:5px; background:#101622; padding:6px; border:1px solid #263247; border-radius:12px; }
    .stTabs [data-baseweb="tab"] { background:transparent; border-radius:8px; padding:9px 12px; color:#aab6cb; font-weight:650; }
    .stTabs [aria-selected="true"] { background:#29264a !important; color:#c8c0ff !important; box-shadow:inset 0 0 0 1px #51478c; }
    [data-testid="stAlert"] { border-radius:11px; background:#151d2c; }
    [data-testid="stExpander"] { border:1px solid #29364b; border-radius:11px; background:rgba(16,22,34,.75); }
    hr { border-color:#273247; margin:1.25rem 0; }
    .site-footer { margin-top:40px; padding:20px 4px 6px; border-top:1px solid #222d40; display:flex; justify-content:space-between; gap:15px; color:#71819b; font-size:11px; }
    .footer-brand { color:#c3cce0; font-weight:700; }
    .footer-note { text-align:right; }
    @media(max-width:768px) {
      .block-container { padding:1rem .8rem 1.5rem; }
      .site-header { padding:12px; align-items:flex-start; }
      .header-right { display:none; }
      .hero { padding:24px 20px; }
      .site-footer { flex-direction:column; }
      .footer-note { text-align:left; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZATION
# ============================================================

init_db()

st.sidebar.markdown(
    """
    <div class="brand-lockup" style="margin:4px 0 18px;">
      <div class="brand-mark" aria-label="Data monitoring logo"><svg viewBox="0 0 48 48" width="29" height="29" role="img" aria-hidden="true"><rect x="5" y="27" width="7" height="14" rx="2" fill="#c8c6ff"/><rect x="16" y="19" width="7" height="22" rx="2" fill="#a8b4ff"/><rect x="27" y="11" width="7" height="30" rx="2" fill="#72e3cf"/><path d="M7 22 L18 14 L29 7 L40 10" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="40" cy="10" r="3" fill="#72e3cf" stroke="#ffffff" stroke-width="1.2"/></svg></div>
      <div><div class="brand-name">DataSentinel AI</div><div class="brand-sub">DATA INTELLIGENCE</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("WORKSPACE NAVIGATION")
st.sidebar.divider()

if "requested_page" in st.session_state:
    st.session_state["active_page"] = st.session_state.pop("requested_page")
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Data Scanner", "Audit History", "About"],
    key="active_page",
)

st.sidebar.divider()
st.sidebar.markdown("**Analysis toolkit**")
st.sidebar.caption("CSV profiling · Validation · ML anomalies")
st.sidebar.markdown("---")
st.sidebar.caption("Your data is processed by this app. Avoid uploading confidential data to a public demo.")


# ============================================================
# HELPERS
# ============================================================

def make_file_fingerprint(file_bytes: bytes) -> str:
    """Return a content-based fingerprint to avoid matching only by filename."""
    return hashlib.sha256(file_bytes).hexdigest()


def calculate_quality_score(df: pd.DataFrame, anomaly_indices) -> float:
    """
    Approximate row-level score based on rows containing missing values,
    duplicate rows after the first occurrence, or flagged anomalies.
    This is an indicator, not a certified data-quality metric.
    """
    total = len(df)
    if total == 0:
        return 0.0

    affected = set(df.index[df.isna().any(axis=1)].tolist())
    affected.update(df.index[df.duplicated(keep="first")].tolist())
    affected.update(set(anomaly_indices or []))

    return round(max(0.0, 100.0 * (1 - len(affected) / total)), 1)


def run_scan(df: pd.DataFrame, source_name: str, fingerprint: str, custom_rules: dict):
    """Run quality checks, anomaly detection, and persist scan metadata."""
    report = run_quality_checks(df)
    validation_issues = validate_dataset(df, rules=custom_rules)
    results = detect_anomalies(df)

    anomaly_mask = (
        results["is_anomaly"]
        .fillna(False)
        .astype(bool)
    )
    anomaly_indices = results.index[anomaly_mask].tolist()
    anomaly_count = int(anomaly_mask.sum())

    quarantine = results.loc[anomaly_mask].copy()
    safe_data = results.loc[~anomaly_mask].copy()

    # Remove exact duplicates from the non-quarantined output.
    safe_data = safe_data.drop_duplicates()

    # Do not include model-internal columns in the business-data downloads.
    internal_columns = ["anomaly_score", "is_anomaly"]
    safe_data = safe_data.drop(columns=internal_columns, errors="ignore")
    quarantine = quarantine.drop(columns=internal_columns, errors="ignore")

    scan_id = save_scan(report, len(df), anomaly_count)
    save_scan_issues(
        scan_id,
        validation_issues,
        action_taken="Flagged for review",
    )

    st.session_state["scan_report"] = report
    st.session_state["scan_results"] = results
    st.session_state["safe_data"] = safe_data
    st.session_state["quarantine"] = quarantine
    st.session_state["validation_issues"] = validation_issues
    st.session_state["source_name"] = source_name
    st.session_state["file_fingerprint"] = fingerprint
    st.session_state["latest_scan_id"] = scan_id
    st.session_state["original_data"] = df.copy()
    st.session_state["anomaly_indices"] = anomaly_indices


def show_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">DATA QUALITY INTELLIGENCE</div>
            <div class="hero-title">Clean data.<br>Confident decisions.</div>
            <div class="hero-copy">
                Discover data quality issues, validate business rules,
                detect unusual records, and create review-ready datasets
                in one intelligent workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_scan_metrics():
    df = st.session_state["original_data"]
    report = st.session_state["scan_report"]
    safe_data = st.session_state["safe_data"]
    quarantine = st.session_state["quarantine"]
    anomaly_indices = st.session_state.get("anomaly_indices", [])

    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    invalid_count = int(sum(report.get("invalid_values", {}).values()))
    quality_score = calculate_quality_score(df, anomaly_indices)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality Score", f"{quality_score}%")
    c2.metric("Missing Values", f"{missing_count:,}")
    c3.metric("Duplicate Rows", f"{duplicate_count:,}")
    c4.metric("Anomalies Flagged", f"{len(quarantine):,}")

    st.caption(
        f"Latest file: {st.session_state['source_name']} · "
        f"Scan ID: {st.session_state['latest_scan_id']} · "
        f"{len(df):,} input records · {len(safe_data):,} clean-output records"
    )

    st.caption(
        "Quality score is an approximate row-level indicator. "
        "It is not a certified quality metric; unusual records may be valid."
    )


def show_results_tabs(df: pd.DataFrame):
    safe_data = st.session_state["safe_data"]
    quarantine = st.session_state["quarantine"]
    validation_issues = st.session_state["validation_issues"]
    results = st.session_state["scan_results"]
    report = st.session_state["scan_report"]

    preview_tab, issues_tab, anomaly_tab, clean_tab, quarantine_tab = st.tabs(
        [
            "📄 Data Preview",
            "⚠️ Validation Issues",
            "🔎 Anomalies",
            "✓ Cleaned Data",
            "📦 Quarantine",
        ]
    )

    with preview_tab:
        st.caption(f"Showing up to 100 of {len(df):,} input records.")
        st.dataframe(
            df.head(100),
            width="stretch",
            hide_index=True,
        )

    with issues_tab:
        if validation_issues.empty:
            st.success("No validation issues detected by the configured rules.")
        else:
            st.dataframe(
                validation_issues,
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download validation issues",
                validation_issues.to_csv(index=False),
                "validation_issues.csv",
                "text/csv",
                key="download_issues",
            )

        with st.expander("Quality-check details"):
            st.json(report)

    with anomaly_tab:
        anomaly_rows = results.loc[
            results["is_anomaly"].fillna(False).astype(bool)
        ]
        if anomaly_rows.empty:
            st.success("No anomalies were flagged by the model.")
        else:
            st.warning(
                f"{len(anomaly_rows):,} records were flagged for review. "
                "An anomaly is unusual, not necessarily incorrect."
            )
            st.dataframe(
                anomaly_rows.head(100),
                width="stretch",
                hide_index=True,
            )

    with clean_tab:
        st.caption(
            f"{len(safe_data):,} rows after excluding flagged anomalies "
            "and removing exact duplicate rows."
        )
        st.dataframe(
            safe_data.head(100),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download cleaned CSV",
            safe_data.to_csv(index=False),
            "cleaned_data.csv",
            "text/csv",
            key="download_clean",
        )

    with quarantine_tab:
        st.caption(f"{len(quarantine):,} rows flagged for review.")
        if quarantine.empty:
            st.success("The quarantine is empty.")
        else:
            st.dataframe(
                quarantine.head(100),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download quarantine CSV",
                quarantine.to_csv(index=False),
                "quarantined_records.csv",
                "text/csv",
                key="download_quarantine",
            )


def render_site_header():
    st.markdown(
        """
        <div class="site-header">
          <div class="brand-lockup">
            <div class="brand-mark" aria-label="Data monitoring logo"><svg viewBox="0 0 48 48" width="29" height="29" role="img" aria-hidden="true"><rect x="5" y="27" width="7" height="14" rx="2" fill="#c8c6ff"/><rect x="16" y="19" width="7" height="22" rx="2" fill="#a8b4ff"/><rect x="27" y="11" width="7" height="30" rx="2" fill="#72e3cf"/><path d="M7 22 L18 14 L29 7 L40 10" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="40" cy="10" r="3" fill="#72e3cf" stroke="#ffffff" stroke-width="1.2"/></svg></div>
            <div>
              <div class="brand-name">DataSentinel AI</div>
              <div class="brand-sub">DATA QUALITY & ANOMALY INTELLIGENCE</div>
            </div>
          </div>
          <div class="header-right">
            <span class="status-pill"><span class="status-dot"></span> SYSTEM READY</span>
            <span>Workspace&nbsp; / &nbsp;Secure analysis</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_intro(eyebrow: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="page-intro">
          <div>
            <div class="eyebrow">{eyebrow}</div>
            <h1 style="margin:0">{title}</h1>
            <div class="intro-copy">{description}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_site_footer():
    st.markdown(
        """
        <div class="site-footer">
          <div><span class="footer-brand">DataSentinel AI</span> &nbsp;·&nbsp; Data quality, made visible.</div>
          <div class="footer-note">Built for transparent data review &nbsp;·&nbsp; Anomalies require human judgment.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dark_charts(issue_data: pd.DataFrame, clean_count: int, quarantine_count: int):
    """Render themed Matplotlib charts with a donut chart and issue breakdown."""
    plt.rcParams.update({
        "text.color": "#dbe4f5",
        "axes.labelcolor": "#9aa9c0",
        "xtick.color": "#9aa9c0",
        "ytick.color": "#b8c4d9",
        "font.family": "DejaVu Sans",
    })

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<div class="panel-title">Issue breakdown</div><div class="panel-sub">Count of quality findings by category</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7.0, 3.25), dpi=130)
        fig.patch.set_facecolor("#101622")
        ax.set_facecolor("#101622")
        if issue_data.empty:
            ax.text(.5, .5, "No issues detected", ha="center", va="center", color="#45d6a0", fontsize=13, transform=ax.transAxes)
            ax.axis("off")
        else:
            plot_data = issue_data.sort_values("Count", ascending=True)
            bars = ax.barh(plot_data["Issue"], plot_data["Count"], color=["#52d6e8", "#8b7cff", "#f5bd68", "#fb7185"][-len(plot_data):], height=.56)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#2b364a")
            ax.tick_params(axis="y", length=0, labelsize=9, pad=9)
            ax.tick_params(axis="x", labelsize=8, colors="#72829b")
            ax.grid(axis="x", color="#263247", linewidth=.7, alpha=.75)
            ax.set_axisbelow(True)
            for bar, value in zip(bars, plot_data["Count"]):
                ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2, f"  {int(value):,}", va="center", ha="left", color="#edf2ff", fontsize=9)
            ax.set_xlim(0, max(1, int(plot_data["Count"].max()) * 1.2))
        fig.tight_layout(pad=1.4)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:
        st.markdown('<div class="panel-title">Record distribution</div><div class="panel-sub">Clean output compared with quarantined records</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.25), dpi=130)
        fig.patch.set_facecolor("#101622")
        ax.set_facecolor("#101622")
        total = int(clean_count + quarantine_count)
        if total == 0:
            ax.text(.5, .5, "No records to display", ha="center", va="center", color="#9aa9c0", fontsize=12, transform=ax.transAxes)
            ax.axis("off")
        else:
            values = [max(0, clean_count), max(0, quarantine_count)]
            colors = ["#45d6a0", "#fb7185"]
            wedges, _ = ax.pie(
                values, startangle=90, counterclock=False, colors=colors,
                wedgeprops={"width": .25, "edgecolor": "#101622", "linewidth": 3},
            )
            ax.text(0, .08, f"{total:,}", ha="center", va="center", color="#f4f7ff", fontsize=21, fontweight="bold")
            ax.text(0, -.18, "TOTAL ROWS", ha="center", va="center", color="#8d9cb4", fontsize=7, fontweight="bold")
            ax.legend(
                wedges, [f"Clean output  {clean_count:,}", f"Quarantined  {quarantine_count:,}"],
                loc="lower center", bbox_to_anchor=(.5, -.12), frameon=False,
                labelcolor="#b8c4d9", fontsize=8, ncol=1,
            )
            ax.axis("equal")
        fig.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# ============================================================
# OVERVIEW
# ============================================================

render_site_header()

if page == "Overview":
    show_hero()

    render_page_intro("COMMAND CENTER", "Workspace overview", "A live view of dataset quality, validation findings and anomaly review.")

    if "scan_report" in st.session_state:
        show_scan_metrics()

        st.markdown("### Data quality analytics")
        df = st.session_state["original_data"]
        report = st.session_state["scan_report"]
        quarantine = st.session_state["quarantine"]
        safe_data = st.session_state["safe_data"]

        issue_data = pd.DataFrame({
            "Issue": ["Missing values", "Duplicate rows", "Invalid values", "Anomalies"],
            "Count": [
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                int(sum(report.get("invalid_values", {}).values())),
                int(len(quarantine)),
            ],
        })
        issue_data = issue_data[issue_data["Count"] > 0]
        render_dark_charts(issue_data, len(safe_data), len(quarantine))

        st.markdown("### Latest scan details")
        show_results_tabs(df)

    else:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Welcome to your data quality workspace</div>
                <div class="feature-copy">
                    Start by uploading a CSV dataset in Data Scanner.
                    DataSentinel AI will profile the data, check validation rules,
                    detect unusual records, and prepare downloadable outputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Open Data Scanner →", type="primary"):
            st.session_state["requested_page"] = "Data Scanner"
            st.rerun()

    st.markdown("### Platform Capabilities")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📊 Data Profiling</div>
                <div class="feature-copy">
                    Inspect missing values, duplicate rows, invalid values,
                    and basic schema information.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🧠 Anomaly Detection</div>
                <div class="feature-copy">
                    Use Isolation Forest to flag unusual records for
                    human review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🧾 Audit History</div>
                <div class="feature-copy">
                    Review saved scan summaries and recorded validation issues.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DATA SCANNER
# ============================================================

elif page == "Data Scanner":
    render_page_intro("DATA WORKSPACE", "Data scanner", "Import a CSV, configure validation rules, and run an explainable quality scan.")

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"],
        key="scanner_upload",
        help="Use a CSV file with a header row.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Ready when you are</div>
                <div class="feature-copy">
                    Upload a CSV to preview the dataset, configure validation rules,
                    run anomaly detection, and download review-ready outputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        file_bytes = uploaded_file.getvalue()
        fingerprint = make_file_fingerprint(file_bytes)

        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except (
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError,
            pd.errors.EmptyDataError,
        ) as exc:
            st.error(f"Unable to read this CSV file: {exc}")
            st.stop()

        if df.empty:
            st.warning("The uploaded CSV contains no data rows.")
            st.stop()

        if len(df.columns) == 0:
            st.error("The uploaded CSV contains no columns.")
            st.stop()

        if df.columns.duplicated().any():
            st.error(
                "Duplicate column names detected. Please rename them "
                "before scanning to avoid ambiguous results."
            )
            st.stop()

        st.markdown("### Dataset Snapshot")
        a, b, c = st.columns(3)
        a.metric("Total Records", f"{len(df):,}")
        b.metric("Total Columns", f"{len(df.columns):,}")
        c.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

        with st.expander("Preview uploaded data", expanded=True):
            st.dataframe(
                df.head(10),
                width="stretch",
                hide_index=True,
            )

        st.markdown("### Configure Validation Rules")
        st.caption(
            "Rules apply to the selected column for this scan. "
            "Leave the rule type as None to skip custom validation."
        )

        custom_rules = {}
        selected_column = st.selectbox(
            "Column",
            ["None"] + df.columns.tolist(),
            key=f"rule_column_{fingerprint[:12]}",
        )

        if selected_column != "None":
            rule_type = st.selectbox(
                "Rule type",
                ["None", "Allowed values", "Numeric range"],
                key=f"rule_type_{fingerprint[:12]}",
            )

            if rule_type == "Allowed values":
                allowed_text = st.text_input(
                    "Allowed values separated by commas",
                    placeholder="Pending, Shipped, Delivered",
                    key=f"allowed_{fingerprint[:12]}",
                )
                if allowed_text.strip():
                    custom_rules[selected_column] = {
                        "allowed": [
                            value.strip()
                            for value in allowed_text.split(",")
                        ]
                    }

            elif rule_type == "Numeric range":
                low, high = st.columns(2)
                with low:
                    min_value = st.number_input(
                        "Minimum allowed value",
                        value=0.0,
                        key=f"min_{fingerprint[:12]}",
                    )
                with high:
                    max_value = st.number_input(
                        "Maximum allowed value",
                        value=1000.0,
                        key=f"max_{fingerprint[:12]}",
                    )

                if min_value <= max_value:
                    custom_rules[selected_column] = {
                        "min": min_value,
                        "max": max_value,
                    }
                else:
                    st.error("Minimum must not exceed maximum.")

        run_clicked = st.button(
            "▶ Run Data Quality Scan",
            type="primary",
            width="stretch",
            key=f"run_scan_{fingerprint[:12]}",
        )

        if run_clicked:
            with st.spinner("Profiling data and detecting anomalies..."):
                try:
                    run_scan(
                        df=df,
                        source_name=uploaded_file.name,
                        fingerprint=fingerprint,
                        custom_rules=custom_rules,
                    )
                    st.success(
                        f"Scan completed successfully. Scan ID: "
                        f"{st.session_state['latest_scan_id']}"
                    )
                except Exception as exc:
                    st.error(f"Scan failed: {exc}")

        # Display results only when they belong to this exact uploaded file.
        if (
            st.session_state.get("file_fingerprint") == fingerprint
            and "scan_report" in st.session_state
        ):
            st.divider()
            st.markdown("## Scan Results")
            show_scan_metrics()
            st.markdown("### Scan analytics")
            report = st.session_state["scan_report"]
            quarantine = st.session_state["quarantine"]
            safe_data = st.session_state["safe_data"]

            issue_data = pd.DataFrame({
                "Issue": ["Missing values", "Duplicate rows", "Invalid values", "Anomalies"],
                "Count": [
                    int(df.isna().sum().sum()),
                    int(df.duplicated().sum()),
                    int(sum(report.get("invalid_values", {}).values())),
                    int(len(quarantine)),
                ],
            })
            issue_data = issue_data[issue_data["Count"] > 0]
            render_dark_charts(issue_data, len(safe_data), len(quarantine))

            st.markdown("### Explore results")
            show_results_tabs(df)


# ============================================================
# AUDIT HISTORY
# ============================================================

elif page == "Audit History":
    render_page_intro("GOVERNANCE", "Audit history", "Trace previous scans, review recorded validation issues, and export audit summaries.")

    history = get_scan_history()

    if history:
        history_df = pd.DataFrame(
            history,
            columns=[
                "Scan ID",
                "Scanned At",
                "Total Records",
                "Missing Values",
                "Duplicate Rows",
                "Invalid Values",
                "Anomalies Detected",
            ],
        )

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Export scan history",
            history_df.to_csv(index=False),
            "scan_history.csv",
            "text/csv",
        )

        scan_ids = history_df["Scan ID"].tolist()
        selected_scan = st.selectbox(
            "Inspect a scan",
            scan_ids,
            key="audit_scan_selector",
        )

        issue_history = get_scan_issues(int(selected_scan))

        if issue_history:
            issue_df = pd.DataFrame(
                issue_history,
                columns=[
                    "Column",
                    "Issue Type",
                    "Issue Count",
                    "Details",
                    "Action Taken",
                ],
            )
            st.markdown(f"### Issues for Scan #{selected_scan}")
            st.dataframe(
                issue_df,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "No detailed validation issues were recorded for this scan."
            )
    else:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">No scan history yet</div>
                <div class="feature-copy">
                    Once you run a scan, its summary and validation issues
                    will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":
    render_page_intro("PLATFORM DETAILS", "About DataSentinel AI", "A transparent workspace for profiling, validation, anomaly review and audit history.")
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">A data quality monitoring workspace</div>
            <div class="feature-copy">
                DataSentinel AI combines rule-based validation, statistical
                anomaly detection, data cleaning, and scan-history reporting
                to help people inspect datasets before using them downstream.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Technology Stack")
    tech1, tech2, tech3 = st.columns(3)
    with tech1:
        st.markdown("**Core**")
        st.write("Python")
        st.write("Pandas")
        st.write("NumPy")
    with tech2:
        st.markdown("**Machine Learning**")
        st.write("Scikit-learn")
        st.write("Isolation Forest")
        st.write("Precision, recall and F1 evaluation")
    with tech3:
        st.markdown("**Application**")
        st.write("Streamlit")
        st.write("SQLite")
        st.write("Pytest")

    st.markdown("### Important Limitations")
    st.markdown(
        """
        - Anomaly detection can produce false positives and false negatives.
        - A flagged record is unusual, not necessarily incorrect.
        - Validation rules and the quality score are not a substitute for
          domain-specific data governance.
        - The SQLite audit database is local to this deployment unless
          persistent storage is configured.
        - This prototype does not yet provide separate user accounts or
          isolated multi-user audit history. Do not use it for confidential
          datasets on a public deployment until that is addressed.
        """
    )

render_site_footer()
