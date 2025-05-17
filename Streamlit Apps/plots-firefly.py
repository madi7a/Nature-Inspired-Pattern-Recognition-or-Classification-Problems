import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Firefly Algorithm history CSV
csv_path = "..\Results\firefly_algorithm_with_diversity.csv" # Ensure this CSV is in the same directory or update the path

st.title("Firefly Algorithm Explorer")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    return df

df = load_data()

# Sidebar controls
run_id = st.sidebar.selectbox("Select Run ID", sorted(df['Run'].unique()))
show_params = st.sidebar.checkbox("Show Parameter Changes", value=True)

# Filter by selected run and sort
run_df = df[df['Run'] == run_id].sort_values(by="Generation")

generations = sorted(run_df['Generation'].unique())
selected_generation = st.slider("Select Generation", min_value=int(min(generations)), max_value=int(max(generations)), value=int(min(generations)))

# Filter for current generation only
gen_df = run_df[run_df['Generation'] == selected_generation]

# Plot Firefly positions (C vs Gamma) colored by Fitness
st.subheader(f"Firefly Positions in Generation {selected_generation} (Run {run_id})")
fig1, ax1 = plt.subplots()
scatter = ax1.scatter(gen_df['C'], gen_df['Gamma'], c=gen_df['Fitness'], cmap='viridis', s=80, edgecolors='black')
cbar = fig1.colorbar(scatter, ax=ax1)
cbar.set_label('Fitness')
ax1.set_xlabel("C")
ax1.set_ylabel("Gamma")
ax1.set_title("Firefly Distribution by Fitness")
ax1.grid(True)
st.pyplot(fig1)

# Line plot: Fitness over generations
st.subheader(f"Best Fitness over Generations for Run {run_id}")
best_fitness_per_gen = run_df.groupby('Generation')['Fitness'].max().reset_index()
fig2, ax2 = plt.subplots()
ax2.plot(best_fitness_per_gen['Generation'], best_fitness_per_gen['Fitness'], marker="o", color="green")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best Fitness")
ax2.set_title("Evolution of Best Fitness")
ax2.grid(True)
st.pyplot(fig2)

# Optional: Parameter changes (diversity, etc.)
if show_params and 'Diversity' in run_df.columns:
    st.subheader("Population Diversity over Generations")
    diversity_per_gen = run_df.groupby('Generation')['Diversity'].mean().reset_index()
    fig3, ax3 = plt.subplots()
    ax3.plot(diversity_per_gen['Generation'], diversity_per_gen['Diversity'], marker='s', color='orange')
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Diversity")
    ax3.set_title("Diversity Trend")
    ax3.grid(True)
    st.pyplot(fig3)

# Show table if user wants
if st.checkbox("Show Raw Data Table"):
    st.dataframe(gen_df.reset_index(drop=True))
