import streamlit as st
import urllib.request
import json
import networkx as nx
import matplotlib.pyplot as plt

# Strict professional design configuration
st.set_page_config(page_title="Causal-DML Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .css-1d391kg, .css-1dp5vir {
        background-color: #262730;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #0078d4;
        color: white;
        border-radius: 4px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #005a9e;
    }
</style>
""", unsafe_allow_html=True)

st.title("Counterfactual Simulation Engine")
st.markdown("This dashboard interfaces with the Causal-DML inference backend to simulate the Conditional Average Treatment Effect (CATE) of retention interventions.")

# Sidebar Configuration
st.sidebar.header("Confounder Configuration")
st.sidebar.markdown("Define the user behavioral and demographic features.")

total_active_days = st.sidebar.number_input("Total Active Days", min_value=0.0, max_value=365.0, value=15.0, step=1.0)
var_daily_listening_time = st.sidebar.number_input("Variance in Daily Listening Time", min_value=0.0, value=300.0, step=10.0)
total_listening_time = st.sidebar.number_input("Total Listening Time (Seconds)", min_value=0.0, value=15000.0, step=500.0)
avg_num_100 = st.sidebar.number_input("Average Full Tracks (100%)", min_value=0.0, value=25.0, step=1.0)

if st.sidebar.button("Execute Simulation"):
    payload = {
        "total_active_days": total_active_days,
        "var_daily_listening_time": var_daily_listening_time,
        "total_listening_time": total_listening_time,
        "avg_num_100": avg_num_100
    }
    
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict_cate", 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            cate = result.get('cate', 0.0)
            
            st.subheader("Intervention Analysis")
            
            # Formulate quantitative metric deltas
            baseline_prob = 0.35  # Assumed static baseline for presentation
            counterfactual_prob = baseline_prob + cate
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Baseline Churn Probability", value=f"{baseline_prob:.2%}")
            with col2:
                st.metric(label="Counterfactual Probability", value=f"{counterfactual_prob:.2%}", delta=f"{cate:.2%}", delta_color="inverse")
            with col3:
                st.metric(label="Conditional Average Treatment Effect", value=f"{cate:.4f}")
                
    except Exception as e:
        st.error(f"Inference Backend Communication Failure: {str(e)}")

st.markdown("---")
st.subheader("Structural Causal Model (DAG)")
st.markdown("Theoretical foundation mapping the causal pathways between confounders, treatment, and outcome.")

# Render Visual Causal Graph (DAG)
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')

G = nx.DiGraph()
confounders = ['Active Days', 'Var Listening', 'Total Listening', 'Avg Tracks']
G.add_nodes_from(confounders)
G.add_node('Intervention (Discount)')
G.add_node('Outcome (Churn)')

for c in confounders:
    G.add_edge(c, 'Intervention (Discount)')
    G.add_edge(c, 'Outcome (Churn)')

G.add_edge('Intervention (Discount)', 'Outcome (Churn)')

pos = {
    'Active Days': (-1.5, 1),
    'Var Listening': (-0.5, 1),
    'Total Listening': (0.5, 1),
    'Avg Tracks': (1.5, 1),
    'Intervention (Discount)': (-0.5, 0),
    'Outcome (Churn)': (0.5, 0)
}

nx.draw(G, pos, ax=ax, with_labels=True, node_color='#262730', font_color='#fafafa', 
        edge_color='#555555', node_size=3500, font_size=10, font_weight='bold', 
        arrowsize=15, edgecolors='#0078d4')

st.pyplot(fig)
