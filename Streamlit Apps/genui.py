import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import os
import kagglehub

# --- Load dataset ---
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("wenruliu/adult-income-dataset")
    for root, dirs, files in os.walk(path):
        if "adult.csv" in files:
            return pd.read_csv(os.path.join(root, "adult.csv"))
    return None

df = load_data()

st.title("Genetic Algorithm for SVM Hyperparameter Tuning")

# Target & Features
target_col = st.selectbox("Select target column:", df.columns, index=df.columns.get_loc("income") if "income" in df.columns else 0)
feature_cols = [col for col in df.columns if col != target_col]
st.markdown("Using all other columns as features:")
st.write(feature_cols)

X = df[feature_cols]
y = df[target_col]

# Preprocessing pipeline
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# GA Parameters input
st.sidebar.header("GA Parameters")
N_POPULATION = st.sidebar.slider("Population size", 10, 100, 30)
N_ITERATIONS = st.sidebar.slider("Generations", 1, 30, 10)
MUTATION_RATE = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.5, 0.05)
MUTATION_STRENGTH = st.sidebar.slider("Mutation strength", 0.01, 0.5, 0.1, 0.01)
MIN_VAL_C = 0.1
MAX_VAL_C = 100.0
MIN_VAL_GAMMA = 0.0001
MAX_VAL_GAMMA = 1.0

# GA operator choices
st.sidebar.header("GA Operators")

selection_method = st.sidebar.selectbox("Selection method", ["Tournament", "Roulette"])
crossover_method = st.sidebar.selectbox("Crossover method", ["Uniform", "Single-point"])
mutation_method = st.sidebar.selectbox("Mutation method", ["Creep", "Swap"])

# --- GA functions ---

def random_chromosome():
    return [random.uniform(MIN_VAL_C, MAX_VAL_C), random.uniform(MIN_VAL_GAMMA, MAX_VAL_GAMMA)]

def fitness(chromosome):
    C, gamma = chromosome
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('svc', SVC(C=C, gamma=gamma))
    ])
    try:
        scores = cross_val_score(model, X, y, cv=5)
        return scores.mean()
    except Exception as e:
        return 0.0

def evaluate_population(pop):
    return [[ind[0], ind[1], fitness(ind)] for ind in pop]

# Selection methods
def tournament_selection(pop, k=3):
    return max(random.sample(pop, k), key=lambda x: x[2])

def roulette_selection(pop):
    max_fitness = sum(ind[2] for ind in pop)
    pick = random.uniform(0, max_fitness)
    current = 0
    for ind in pop:
        current += ind[2]
        if current >= pick:
            return ind
    return pop[-1]

def select(pop):
    if selection_method == "Tournament":
        return tournament_selection(pop)
    elif selection_method == "Roulette":
        return roulette_selection(pop)

# Crossover methods
def uniform_crossover(p1, p2):
    c = p1[0] if random.random() < 0.5 else p2[0]
    g = p1[1] if random.random() < 0.5 else p2[1]
    return [c, g]

def single_point_crossover(p1, p2):
    if random.random() < 0.5:
        return [p1[0], p2[1]]
    else:
        return [p2[0], p1[1]]

def crossover(p1, p2):
    if crossover_method == "Uniform":
        return uniform_crossover(p1, p2)
    elif crossover_method == "Single-point":
        return single_point_crossover(p1, p2)

# Mutation methods
def creep_mutation(ind):
    c, g = ind
    if random.random() < MUTATION_RATE:
        c += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
        g += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
    return [
        min(max(c, MIN_VAL_C), MAX_VAL_C),
        min(max(g, MIN_VAL_GAMMA), MAX_VAL_GAMMA)
    ]

def swap_mutation(ind):
    # Swap mutation: swap C and gamma with some probability
    if random.random() < MUTATION_RATE:
        return [ind[1], ind[0]]
    else:
        return ind

def mutate(ind):
    if mutation_method == "Creep":
        return creep_mutation(ind)
    elif mutation_method == "Swap":
        return swap_mutation(ind)

# --- Run GA ---
def run_genetic_algorithm():
    population = [random_chromosome() for _ in range(N_POPULATION)]
    population = evaluate_population(population)
    bests = []

    for gen in range(N_ITERATIONS):
        new_pop = []
        while len(new_pop) < N_POPULATION:
            p1 = select(population)
            p2 = select(population)
            child = crossover(p1, p2)
            child = mutate(child)
            child_fit = fitness(child)
            new_pop.append([child[0], child[1], child_fit])
        population = new_pop
        best = max(population, key=lambda x: x[2])
        bests.append((gen, best[0], best[1], best[2]))
        st.text(f"Generation {gen} - Best fitness: {best[2]:.4f}")
    return bests

# --- Run button ---
if st.button("Run Genetic Algorithm"):
    with st.spinner("Running genetic algorithm..."):
        best_results = run_genetic_algorithm()
        df_results = pd.DataFrame(best_results, columns=["Generation", "C", "Gamma", "Fitness"])
        st.success("GA finished!")
        st.dataframe(df_results)

        # Plot fitness curve
        fig, ax = plt.subplots()
        ax.plot(df_results["Generation"], df_results["Fitness"], marker='o')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title("GA Best Fitness over Generations")
        st.pyplot(fig)
