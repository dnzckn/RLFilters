
import torch
import numpy as np
from src.model import EM_Emulator
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_PATH = "/Users/deniz/Documents/GitHub/RLFilters/models/em_emulator.pth"
GRID_SIZE = 16
NUM_FREQ_POINTS = 100
POPULATION_SIZE = 100
NUM_GENERATIONS = 50
MUTATION_RATE = 0.01

# --- 1. Load Trained Model ---
def load_model(model_path, grid_size, num_freq_points):
    model = EM_Emulator(grid_size=grid_size, num_freq_points=num_freq_points)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# --- 2. Define Target S-parameter Response ---
def get_target_response(num_freq_points):
    # Simple bandpass filter centered at 50
    freq_axis = np.arange(num_freq_points)
    target_s_params = np.exp(-((freq_axis - 50) / 10)**2)
    return torch.from_numpy(target_s_params).float()

# --- 3. Genetic Algorithm ---
def initialize_population(population_size, grid_size):
    return np.random.randint(0, 2, size=(population_size, grid_size, grid_size))

def calculate_fitness(population, model, target_response):
    fitness_scores = []
    for individual in population:
        layout_tensor = torch.from_numpy(individual).unsqueeze(0).unsqueeze(0).float()
        predicted_s_params = model(layout_tensor)
        loss = torch.mean((predicted_s_params - target_response)**2)
        fitness_scores.append(1 / (loss.item() + 1e-6))
    return np.array(fitness_scores)

def selection(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)[::-1]
    return population[sorted_indices[:POPULATION_SIZE // 2]]

def crossover(parents):
    offspring = []
    for _ in range(POPULATION_SIZE):
        parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
        crossover_point = np.random.randint(1, GRID_SIZE * GRID_SIZE -1)
        child = np.concatenate((parent1.flatten()[:crossover_point], parent2.flatten()[crossover_point:])).reshape(GRID_SIZE, GRID_SIZE)
        offspring.append(child)
    return np.array(offspring)

def mutate(population, mutation_rate):
    for i in range(len(population)):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if np.random.rand() < mutation_rate:
                    population[i, r, c] = 1 - population[i, r, c]
    return population

# --- 4. Main Optimization Loop ---
def optimize():
    model = load_model(MODEL_PATH, GRID_SIZE, NUM_FREQ_POINTS)
    target_response = get_target_response(NUM_FREQ_POINTS)

    population = initialize_population(POPULATION_SIZE, GRID_SIZE)

    print("Starting optimization...")
    for generation in range(NUM_GENERATIONS):
        fitness_scores = calculate_fitness(population, model, target_response)
        parents = selection(population, fitness_scores)
        population = crossover(parents)
        population = mutate(population, MUTATION_RATE)

        if (generation + 1) % 10 == 0:
            print(f"Generation {generation+1}/{NUM_GENERATIONS}, Best Fitness: {np.max(fitness_scores):.4f}")

    # --- 5. Get and Display Best Result ---
    final_fitness_scores = calculate_fitness(population, model, target_response)
    best_individual_index = np.argmax(final_fitness_scores)
    best_layout = population[best_individual_index]
    
    # Get the predicted response for the best layout
    best_layout_tensor = torch.from_numpy(best_layout).unsqueeze(0).unsqueeze(0).float()
    best_predicted_s_params = model(best_layout_tensor).detach().numpy().flatten()

    print("Optimization finished.")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(best_layout, cmap='gray')
    plt.title("Optimized Filter Layout")

    plt.subplot(1, 2, 2)
    plt.plot(target_response.numpy(), label="Target Response")
    plt.plot(best_predicted_s_params, label="Predicted Response")
    plt.title("S-Parameter Response")
    plt.xlabel("Frequency Points")
    plt.ylabel("S-Parameter Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("demo_figures/optimized_filter.png")

if __name__ == "__main__":
    optimize()
