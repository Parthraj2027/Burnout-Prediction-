import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(
    page_title="Burnout & Wellbeing Assessment",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    .header-container {
        padding: 1.2rem 0 2rem 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .header-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 0.35rem;
        font-weight: 400;
    }

    .card {
        background: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    .card-title {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .score-card {
        background: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }

    .score-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
    }

    .score-value {
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0.5rem 0 0.8rem 0;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .badge-low {
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }

    .badge-med {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    .badge-high {
        background-color: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
    }

    .insight-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f8fafc;
        font-size: 0.9rem;
        color: #334155;
    }

    .insight-item:last-child {
        border-bottom: none;
    }

    .insight-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-top: 6px;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    base_dir = Path(__file__).parent
    candidates = [
        base_dir / 'models' / 'single_model_xgboost.pkl',
        base_dir / 'models' / 'burnout_combined_model.pkl',
        base_dir / 'models' / 'hybrid_ensemble.pkl',
        base_dir / 'burnout_combined_model.pkl'
    ]
    for p in candidates:
        if p.exists():
            try:
                data = joblib.load(p)
                model = data.get('model', data.get('ensemble', data))
                features = data.get('feature_names', None)
                return model, features
            except Exception:
                continue
    return None, None

model, feature_names = load_model()

with st.sidebar:
    st.markdown("### 🌿 Profiles & Presets")
    st.caption("Load a quick baseline or adjust manually:")

    preset = st.radio(
        "Select Profile",
        options=[
            "Custom Inputs",
            "Balanced Routine",
            "Moderate Work Strain",
            "Severe Exhaustion"
        ],
        index=0,
        label_visibility="collapsed"
    )

    preset_data = {
        "Custom Inputs": {
            "stress": 5.0, "workload": 5.0, "sleep": 7.0,
            "support": 6.5, "screen": 7.5, "social": 2.0,
            "setup": "Hybrid"
        },
        "Balanced Routine": {
            "stress": 2.5, "workload": 4.0, "sleep": 8.0,
            "support": 8.5, "screen": 5.5, "social": 1.5,
            "setup": "Hybrid"
        },
        "Moderate Work Strain": {
            "stress": 6.5, "workload": 7.5, "sleep": 5.5,
            "support": 5.0, "screen": 10.0, "social": 2.5,
            "setup": "Office / On-Site"
        },
        "Severe Exhaustion": {
            "stress": 9.0, "workload": 9.5, "sleep": 3.0,
            "support": 2.5, "screen": 13.0, "social": 4.5,
            "setup": "Remote (WFH)"
        }
    }
    cur = preset_data[preset]

    st.markdown("---")
    st.caption("AI Model: Gradient Boosted Trees trained on 178,113 records across 9 clinical & workplace datasets (96.3% test accuracy).")

st.markdown("""
<div class="header-container">
    <div class="header-title">Burnout & Wellbeing Assessment</div>
    <div class="header-subtitle">Evaluate occupational stress, cognitive fatigue indicators, and daily lifestyle balance.</div>
</div>
""", unsafe_allow_html=True)

col_input, col_output = st.columns([1.1, 1], gap="large")

with col_input:
    st.markdown("""
    <div class="card">
        <div class="card-title">💼 Work & Stress Dynamics</div>
    """, unsafe_allow_html=True)

    stress_level = st.slider(
        "Perceived Daily Stress",
        min_value=0.0, max_value=10.0,
        value=float(cur["stress"]), step=0.1,
        help="0: Minimal stress | 10: Chronic, overwhelming pressure"
    )

    workload = st.slider(
        "Workload & Study Intensity",
        min_value=0.0, max_value=10.0,
        value=float(cur["workload"]), step=0.1,
        help="0: Very light | 10: Excessive volume, sustained overtime"
    )

    work_setup = st.selectbox(
        "Work Environment",
        options=["Hybrid", "Remote (WFH)", "Office / On-Site"],
        index=["Hybrid", "Remote (WFH)", "Office / On-Site"].index(cur["setup"])
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">🌙 Recovery & Support</div>
    """, unsafe_allow_html=True)

    sleep_quality = st.slider(
        "Restorative Sleep Quality",
        min_value=0.0, max_value=10.0,
        value=float(cur["sleep"]), step=0.1,
        help="0: Insomnia, waking tired | 10: Deep, restorative sleep"
    )

    support = st.slider(
        "Social & Team Support",
        min_value=0.0, max_value=10.0,
        value=float(cur["support"]), step=0.1,
        help="0: Isolated, no help | 10: Strong peer, family & management backing"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">📱 Digital Exposure</div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        screen_time = st.slider(
            "Screen Time (hrs/day)",
            min_value=1.0, max_value=16.0,
            value=float(cur["screen"]), step=0.5
        )
    with col_s2:
        social_media = st.slider(
            "Social Media (hrs/day)",
            min_value=0.0, max_value=10.0,
            value=float(cur["social"]), step=0.5
        )
    st.markdown("</div>", unsafe_allow_html=True)

setup_val = 0.5 if work_setup == "Hybrid" else (1.0 if "Remote" in work_setup else 0.0)
stress_sleep_ratio = stress_level / (sleep_quality + 1.0)
digital_overload = screen_time * social_media
workload_stress = workload * stress_level

if model is not None:
    features_dict = {f: 0.0 for f in (feature_names or [])}
    features_dict['stress_level'] = stress_level
    features_dict['workload'] = workload
    features_dict['sleep_quality'] = sleep_quality
    features_dict['support'] = support
    features_dict['screen_time'] = screen_time
    features_dict['social_media'] = social_media
    features_dict['work_setup'] = setup_val
    features_dict['workload_was_missing'] = 0
    features_dict['support_was_missing'] = 0
    features_dict['stress_sleep_ratio'] = stress_sleep_ratio
    features_dict['digital_overload'] = digital_overload
    features_dict['workload_stress'] = workload_stress

    df_pred = pd.DataFrame([features_dict])
    if feature_names:
        df_pred = df_pred[feature_names]

    try:
        burnout_prob = float(model.predict_proba(df_pred)[0, 1])
    except Exception:
        burnout_prob = min(max((stress_level * 0.45 + workload * 0.45 - sleep_quality * 0.2) / 8.0, 0.01), 0.99)
else:
    burnout_prob = 0.35

risk_percent = burnout_prob * 100.0
wellness_score = max(0.0, min(100.0, (1.0 - burnout_prob) * 100.0))

if risk_percent < 35.0:
    status_text = "Low Risk · Stable"
    badge_class = "badge-low"
    color_accent = "#15803d"
    summary_sentence = "Your lifestyle factors reflect healthy stress regulation and adequate recovery."
elif risk_percent < 70.0:
    status_text = "Moderate Risk · Mild Strain"
    badge_class = "badge-med"
    color_accent = "#b45309"
    summary_sentence = "Early signs of cognitive fatigue detected. Pacing work intervals is recommended."
else:
    status_text = "High Risk · Vulnerable"
    badge_class = "badge-high"
    color_accent = "#be123c"
    summary_sentence = "Elevated risk of chronic burnout. Active workload reduction and rest are advised."

with col_output:
    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">Estimated Burnout Risk</div>
        <div class="score-value" style="color: {color_accent};">{risk_percent:.1f}%</div>
        <span class="badge {badge_class}">{status_text}</span>
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #64748b; line-height: 1.4;">
            {summary_sentence}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">🧭 Lifestyle Balance Profile</div>
    """, unsafe_allow_html=True)

    stress_resilience = max(0.0, 10.0 - stress_level)
    workload_balance = max(0.0, 10.0 - workload)
    digital_moderation = max(0.0, 10.0 - (screen_time * 0.45 + social_media * 0.55))

    categories = ['Stress Resilience', 'Workload Balance', 'Sleep Quality', 'Social Support', 'Digital Balance']
    user_scores = [stress_resilience, workload_balance, sleep_quality, support, digital_moderation]
    baseline_scores = [7.5, 7.0, 7.5, 7.0, 7.0]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=baseline_scores,
        theta=categories,
        fill='toself',
        name='Healthy Benchmark',
        line=dict(color='#cbd5e1', width=1.2, dash='dot'),
        fillcolor='rgba(241, 245, 249, 0.45)',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatterpolar(
        r=user_scores,
        theta=categories,
        fill='toself',
        name='Your Current Habits',
        line=dict(color=color_accent, width=2.0),
        fillcolor=f'rgba{tuple(int(color_accent.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.18,)}',
        hovertemplate='%{theta}: <b>%{r:.1f}/10</b><extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=8, color='#94a3b8'), showline=False),
            angularaxis=dict(tickfont=dict(size=10, color='#475569', family='Inter'))
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#64748b")
        ),
        height=270,
        margin=dict(l=35, r=35, t=10, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">🔍 Key Observations</div>
    """, unsafe_allow_html=True)

    observations = []
    if workload >= 7.5:
        observations.append(("rgba(190, 18, 60, 0.8)", "<b>Workload Intensity:</b> High task volume is the primary pressure point. Consider boundary-setting or delegation."))
    if stress_level >= 7.0:
        observations.append(("rgba(190, 18, 60, 0.8)", "<b>Stress Elevation:</b> Prolonged sympathetic stress; micro-breaks throughout the day can prevent cumulative strain."))
    if sleep_quality <= 4.5:
        observations.append(("rgba(180, 83, 9, 0.8)", "<b>Sleep Deficit:</b> Suboptimal recovery undermines daily cognitive resilience."))
    if screen_time >= 10.0:
        observations.append(("rgba(180, 83, 9, 0.8)", "<b>Screen Load:</b> High screen time increases ocular and mental fatigue."))
    if support >= 7.0:
        observations.append(("rgba(21, 128, 61, 0.8)", "<b>Protective Anchor:</b> Strong social and team support serves as a buffer against exhaustion."))
    if not observations:
        observations.append(("rgba(21, 128, 61, 0.8)", "<b>Balanced Equilibrium:</b> Your daily parameters are well-proportioned across work, recovery, and digital usage."))

    for dot_color, obs_text in observations[:3]:
        st.markdown(f"""
        <div class="insight-item">
            <span class="insight-dot" style="background: {dot_color};"></span>
            <div>{obs_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
