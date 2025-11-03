import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Personalized PK (One-Compartment)")

# ---- helpers ----
def oral_one_comp_conc(t, dose_times, D, F, ka, ke, V):
    t = np.asarray(t)
    C = np.zeros_like(t, dtype=float)
    coef = F * D * ka / (V * max(1e-12, (ka - ke)))
    for ti in dose_times:
        dt = t - ti
        mask = dt >= 0
        C[mask] += coef * (np.exp(-ke * dt[mask]) - np.exp(-ka * dt[mask]))
    return C

def pk_metrics(t, C):
    idx = int(np.argmax(C))
    Cmax, Tmax = float(C[idx]), float(t[idx])
    AUC = float(np.trapz(C, t))
    return Cmax, Tmax, AUC

# ---- sidebar inputs ----
st.sidebar.header("Patient & Covariates")
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=70.0, step=1.0)
age_years = st.sidebar.number_input("Age (years)", min_value=0.0, value=40.0, step=1.0)
CL_L_h   = st.sidebar.number_input("Clearance CL (L/h)", min_value=0.1, value=18.0, step=0.5)

st.sidebar.header("Absorption & Distribution")
ka_h     = st.sidebar.number_input("Absorption rate ka (1/h)", min_value=0.01, value=1.4, step=0.05)
F        = st.sidebar.slider("Bioavailability F", min_value=0.05, max_value=0.99, value=0.88, step=0.01)
Vd_per_kg= st.sidebar.number_input("Volume per kg (L/kg)", min_value=0.1, value=0.9, step=0.05)

st.sidebar.header("Dosing Regimen")
dose_mg  = st.sidebar.number_input("Dose (mg)", min_value=1.0, value=1000.0, step=50.0)
tau_h    = st.sidebar.number_input("Interval τ (h)", min_value=1.0, value=8.0, step=1.0)
n_doses  = st.sidebar.number_input("Number of doses", min_value=1, value=3, step=1)
tlag_h   = st.sidebar.number_input("Absorption lag (h)", min_value=0.0, value=0.0, step=0.1)

st.sidebar.header("Simulation")
t_end_h  = st.sidebar.number_input("End time (h)", min_value=1.0, value=24.0, step=1.0)
dt_h     = st.sidebar.number_input("Time step (h)", min_value=0.01, value=0.05, step=0.01)

# ---- derived params ----
V_L  = Vd_per_kg * weight_kg
ke_h = CL_L_h / max(1e-12, V_L)
t    = np.arange(0.0, t_end_h + dt_h, dt_h)
dose_times = np.array([i * tau_h for i in range(int(n_doses))]) + tlag_h

# ---- simulate ----
C = oral_one_comp_conc(t, dose_times, D=dose_mg, F=F, ka=ka_h, ke=ke_h, V=V_L)
Cmax, Tmax, AUC = pk_metrics(t, C)

# ---- render (single column) ----
st.markdown("### Concentration–Time Profile (One-Compartment, Oral)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=C, mode="lines", name="Concentration (mg/L)"))
for dt in dose_times:
    fig.add_vline(x=float(dt), line_dash="dot")
fig.update_layout(
    xaxis_title="Time (h)", yaxis_title="Concentration (mg/L)",
    template="plotly_white",
    title=f"Dose={dose_mg:.0f} mg, τ={tau_h:.0f} h, n={int(n_doses)}"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Parameters")
st.write(
    f"- **Age**: {age_years:.0f} y\n"
    f"- **Weight**: {weight_kg:.1f} kg\n"
    f"- **CL**: {CL_L_h:.2f} L/h\n"
    f"- **V**: {V_L:.2f} L (={Vd_per_kg:.2f} L/kg × {weight_kg:.1f} kg)\n"
    f"- **ke**: {ke_h:.3f} 1/h\n"
    f"- **ka**: {ka_h:.3f} 1/h\n"
    f"- **F**: {F:.2f}"
)

st.markdown("---")
st.markdown("### Outputs")
st.write(
    f"- **Cmax**: {Cmax:.3f} mg/L\n"
    f"- **Tmax**: {Tmax:.2f} h\n"
    f"- **AUC**: {AUC:.2f} mg·h/L"
)

st.caption(
    "Model: one-compartment, oral first-order absorption; "
    "C(t)=Σ F·D·ka/[V(ka−ke)]·(e^(−ke(t−ti)) − e^(−ka(t−ti))) for t ≥ ti. "
    "Age is displayed only; CL is user-provided; V scales with weight."
)



