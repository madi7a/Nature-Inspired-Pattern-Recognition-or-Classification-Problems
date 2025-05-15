import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("Best_Per_Generation_30_Runs.csv")

st.title("Genetic Algorithm Run Explorer")

# Sidebar inputs
run_id = st.sidebar.selectbox("Select Run ID", sorted(df['run'].unique()))
show_params = st.sidebar.checkbox("Show Parameter Changes", value=True)

# Filter the dataframe for the selected run
run_df = df[df['run'] == run_id].sort_values(by="generation")

# Line plot: Fitness over generations
st.subheader(f"Fitness over Generations for Run {run_id}")
fig, ax = plt.subplots()
ax.plot(run_df["generation"], run_df["fitness"], marker="o", label="Fitness", color="green")
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")
ax.grid(True)
st.pyplot(fig)

# Parameter plots
if show_params:
    st.subheader("Parameter Evolution")
    
    fig2, ax2 = plt.subplots()
    ax2.plot(run_df["generation"], run_df["C"], marker="x", label="C", color="blue")
    ax2.set_ylabel("C Value", color="blue")
    ax2.set_xlabel("Generation")

    ax3 = ax2.twinx()
    ax3.plot(run_df["generation"], run_df["gamma"], marker="^", label="Gamma", color="red")
    ax3.set_ylabel("Gamma Value", color="red")

    fig2.tight_layout()
    st.pyplot(fig2)

# Show table if user wants
if st.checkbox("Show Raw Data Table"):
    st.dataframe(run_df)
