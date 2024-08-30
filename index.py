pip install numpy scipy deap matplotlib
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Define the optimization problem
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))  # Minimize distance, fuel, and safety risks
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize the genetic algorithm components
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 100)  # Random float between 0 and 100 for route segments
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)  # Create individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Create population
def evaluate(individual):
    # Simulate calculations (replace with real computations)
    distance = np.sum(individual)  # Total distance (proxy for travel time)
    fuel_consumption = np.mean(individual)  # Average fuel consumption rate
    safety_risk = np.max(individual)  # Maximum risk factor (e.g., high wave heights, currents)
    
    return distance, fuel_consumption, safety_risk
toolbox.register("evaluate", evaluate)  # Evaluation function
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover operator
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)  # Mutation operator
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection operator
# Parameters
population_size = 100
number_of_generations = 50
crossover_probability = 0.5
mutation_probability = 0.2

# Create initial population
population = toolbox.population(n=population_size)

# Run the Genetic Algorithm
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)
stats.register("avg", np.mean, axis=0)

algorithms.eaSimple(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, 
                    ngen=number_of_generations, stats=stats, verbose=True)

# Extract and display the best individual
best_individual = tools.selBest(population, 1)[0]
print("Best Individual: ", best_individual)
print("Fitness: ", evaluate(best_individual))

# Plotting the results
gen = list(range(number_of_generations))
fits = np.array([ind.fitness.values for ind in population])

plt.figure(figsize=(12, 6))

# Plot distance
plt.subplot(3, 1, 1)
plt.plot(gen, [stats.values[0][0] for _ in gen], label="Min Distance")
plt.plot(gen, [stats.values[1][0] for _ in gen], label="Avg Distance")
plt.xlabel("Generation")
plt.ylabel("Distance")
plt.legend()

# Plot fuel consumption
plt.subplot(3, 1, 2)
plt.plot(gen, [stats.values[0][1] for _ in gen], label="Min Fuel Consumption")
plt.plot(gen, [stats.values[1][1] for _ in gen], label="Avg Fuel Consumption")
plt.xlabel("Generation")
plt.ylabel("Fuel Consumption")
plt.legend()

# Plot safety risk
plt.subplot(3, 1, 3)
plt.plot(gen, [stats.values[0][2] for _ in gen], label="Min Safety Risk")
plt.plot(gen, [stats.values[1][2] for _ in gen], label="Avg Safety Risk")
plt.xlabel("Generation")
plt.ylabel("Safety Risk")
plt.legend()

plt.tight_layout()
plt.show()
