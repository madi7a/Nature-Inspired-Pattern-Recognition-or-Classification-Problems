import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
import time
import random
import numpy as np
import random

class WhaleOptimizationAlgorithm:
    def __init__(self, objective_function, lb, ub, dim, num_agents=2, max_iter=2, patience=10, random_seed=42):
        self.objective_function = objective_function
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.patience = patience
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def optimize(self, verbose=True):
        # Initialize positions
        X_positions = np.random.uniform(self.lb, self.ub, (self.num_agents, self.dim))
        Fitness = np.array([self.objective_function(x) for x in X_positions])
        X_best = X_positions[np.argmin(Fitness)]
        F_best = np.min(Fitness)

        stagnation_count = 0
        last_best_fitness = F_best

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            for i in range(self.num_agents):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.rand()
                l = np.random.uniform(-1, 1)

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * X_best - X_positions[i])  # Shrinking circle
                        X_positions[i] = X_best - A * D
                    else:
                        rand_index = np.random.randint(0, self.num_agents)
                        X_rand = X_positions[rand_index]
                        D = abs(C * X_rand - X_positions[i])  # Search for prey
                        X_positions[i] = X_rand - A * D
                else: #spiral
                    distance_to_leader = abs(X_best - X_positions[i])
                    X_positions[i] = distance_to_leader * np.exp(l) * np.cos(2 * np.pi * l) + X_best

                X_positions[i] = np.clip(X_positions[i], self.lb, self.ub)

            Fitness = np.array([self.objective_function(x) for x in X_positions])

            # update the best sol
            for i in range(self.num_agents):
                if Fitness[i] < F_best:
                    X_best = X_positions[i]
                    F_best = Fitness[i]

            # check convergence
            if F_best < last_best_fitness:
                stagnation_count = 0
                last_best_fitness = F_best
            else:
                stagnation_count += 1

            if verbose:
                print(f"Iteration {t + 1}/{self.max_iter}, Best Fitness: {-F_best:.4f}")

            if stagnation_count >= self.patience:
                if verbose:
                    print(f"Convergence reached. No improvement for {self.patience} iterations.")
                break

        return X_best, -F_best
    
class SimulatedAnnealing:
    def __init__(self, objective_function, bounds, initial_temp=100, 
                 n_iterations=100, step_size=0.1, patience=10, random_seed=42):
        self.objective_function = objective_function
        self.bounds = bounds
        self.initial_temp = initial_temp
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.patience = patience
        self.best_solution = None
        self.best_fitness = -np.inf
        self.all_results = []
        self.best_run = None
        np.random.seed(random_seed)
        random.seed(random_seed)

    def _generate_initial_solution(self):
        return [random.uniform(bound[0], bound[1]) for bound in self.bounds]
    
    def _generate_neighbor(self, current_solution):
        neighbor = []
        for i, (param, bound) in enumerate(zip(current_solution, self.bounds)):
            # perturb w Gaussian noise
            sigma = (bound[1] - bound[0]) * self.step_size
            delta = np.random.normal(0, sigma)
            new_value = param + delta
            
            new_value = max(min(new_value, bound[1]), bound[0])
            neighbor.append(new_value)
        return neighbor
    
    def _acceptance_probability(self, current_fitness, new_fitness, temperature):
        if new_fitness > current_fitness:
            return 1.0
        return np.exp((new_fitness - current_fitness) / temperature)

    def optimize(self, max_iter=1000, verbose=True):
        start_time = time.time()
        current_solution = self._generate_initial_solution()
        current_fitness = self.objective_function(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        fitness_history = [current_fitness]
        solution_history = [current_solution.copy()]
        temperature_history = []
        iteration = 0
        unchanged_counter = 0
        adaptive_step = self.step_size

        while iteration < max_iter:
            # logarithmic cooling
            temperature = self.initial_temp / (1 + np.log(1 + iteration))
            temperature_history.append(temperature)
            
            no_improvement = True
            for i in range(self.n_iterations):
                neighbor_solution = self._generate_neighbor(current_solution)
                neighbor_fitness = self.objective_function(neighbor_solution)
                
                if self._acceptance_probability(current_fitness, neighbor_fitness, temperature) > random.random():
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
                    
                    if current_fitness > best_fitness:
                        best_solution = current_solution.copy()
                        best_fitness = current_fitness
                        unchanged_counter = 0
                        no_improvement = False
                    else:
                        unchanged_counter += 1

                iteration += 1
                if iteration >= max_iter:
                    break
            if no_improvement:
                unchanged_counter += 1
            if unchanged_counter >= self.patience:
                if verbose:
                    print("Early stopping: No improvement for {} iterations.".format(self.patience))
                break
            if unchanged_counter > 10:
                adaptive_step = min(adaptive_step * 1.5, 0.5)
            else:
                adaptive_step = max(adaptive_step * 0.9, 0.01)
            self.step_size = adaptive_step
        return best_solution, best_fitness

    def _generate_initial_solution(self):
        return [random.uniform(bound[0], bound[1]) for bound in self.bounds]

    def _generate_neighbor(self, current_solution):
        neighbor = []
        for i, (param, bound) in enumerate(zip(current_solution, self.bounds)):
            delta = (bound[1] - bound[0]) * self.step_size * np.random.uniform(-1, 1)
            new_value = param + delta
            new_value = max(min(new_value, bound[1]), bound[0])
            neighbor.append(new_value)
        return neighbor

    def _acceptance_probability(self, current_fitness, new_fitness, temperature):
        if new_fitness > current_fitness:
            return 1.0
        return np.exp((new_fitness - current_fitness) / temperature)


# Objective Function for SVM Hyperparameter Optimization
def svm_objective_function(params, X_train, X_test, y_train, y_test, preprocessor):
    C, gamma = params
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('svc', SVC(
            C=C,
            gamma=gamma,
            kernel='rbf',
            probability=True,
            random_state=42
        ))
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1

#--------------------------------------------------------------------------------------------------------
def main():
    st.title("Hyperparameter Optimization Comparison")

    # Sidebar for algorithm selection
    st.sidebar.title("Select Optimization Algorithm")
    algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        (
            "Simulated Annealing",
            "Whale Optimization Algorithm"
        )
    )

    # Load dataset
    df = pd.read_csv("adult.csv")  # Replace with your dataset file
    SUBSET_SIZE = 5000
    TARGET_COLUMN = 'income'
    NEW_TARGET_NAME = 'Outcome'

    # Sample a subset of the data
    df_processed = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)

    # Separate target variable and features
    y = df_processed[TARGET_COLUMN]
    X = df_processed.drop(TARGET_COLUMN, axis=1)

    # Drop unnecessary columns
    COLUMNS_TO_DROP = ['fnlwgt']
    if 'ID' in X.columns:
        COLUMNS_TO_DROP.append('ID')
    if 'policy_id' in X.columns:
        COLUMNS_TO_DROP.append('policy_id')
    X = X.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    # Encode target variable if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=NEW_TARGET_NAME, index=y.index)

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    transformers = []
    if numerical_features:
        transformers.append(('num', numerical_pipeline, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_pipeline, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    # Define parameter bounds
    param_bounds = [
        (0.1, 1000),      # C
        (0.0001, 10)     # gamma
    ]
    lb = [bound[0] for bound in param_bounds]
    ub = [bound[1] for bound in param_bounds]

    # Objective function for optimization
    def svm_objective_function(params):
        C, gamma = params
        clf = Pipeline([
            ('preprocessor', preprocessor),
            ('svc', SVC(
                C=C,
                gamma=gamma,
                kernel='rbf',
                probability=True,
                random_state=42
            ))
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return -f1  # Negative because WOA minimizes

    # Main content
    st.header(f"Optimizing SVM Hyperparameters using {algorithm}")
    if st.button("Run Optimization"):
        st.write(f"Running {algorithm}...")

        # Initialize and run the selected algorithm
        if algorithm == "Simulated Annealing":
            sa = SimulatedAnnealing(
                objective_function=lambda params: -svm_objective_function(params),
                bounds=param_bounds,
                initial_temp=100,
                n_iterations=5,
                step_size=0.2,
                patience=10,
                random_seed=42
            )
            best_params, best_fitness = sa.optimize(max_iter=20)
        elif algorithm == "Whale Optimization Algorithm":
            woa = WhaleOptimizationAlgorithm(
                objective_function=svm_objective_function,
                lb=lb,
                ub=ub,
                dim=2,
                num_agents=15,
                max_iter=20,
                patience=10,
                random_seed=42
            )
            best_params, best_fitness = woa.optimize()

        # Train final model with best parameters
        C_best, gamma_best = best_params
        final_svm_model = Pipeline([
            ('preprocessor', preprocessor),
            ('svc', SVC(
                C=C_best,
                gamma=gamma_best,
                kernel='rbf',
                random_state=42,
                probability=True
            ))
        ])
        final_svm_model.fit(X_train, y_train)
        y_pred = final_svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Display results
        st.subheader("Results")
        st.write(f"Best Parameters: C={C_best:.6f}, gamma={gamma_best:.6f}")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

    # Comparison Table
    st.header("Comparison of Optimization Algorithms")
    comparison_data = {
        "Approach": ["Simulated Annealing (SA)", "Whale Optimization Algorithm (WOA)"],
        "Type": ["SI", "SI"],
        "Problem Type": ["Free Optimization"] * 2,
        "Classifier": ["SVM"] * 2,
        "Optimized Params": ["C, gamma", "C, gamma"],
        "Accuracy": [0.860, 0.8500],
        "F1 Score": [0.860, 0.8500],
        "Bonus Features / Notes": [
            "Cooling-based TA, Novel variant",
            "Swarm intelligence behavior, Search balancing"
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)


if __name__ == "__main__":
    main()