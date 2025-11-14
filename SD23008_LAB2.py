import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# =======================
# GA PARAMETERS
# =======================
POP_SIZE = 300
CHROM_LEN = 80
TARGET_ONES = 50
MAX_FITNESS = 80
N_GENERATIONS = 50

TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN

# =======================
# GA FUNCTIONS
# =======================
def fitness(individual):
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)

def init_population(pop_size, chrom_len):
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)

def tournament_selection(pop, fits, k):
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()

def single_point_crossover(p1, p2):
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(individual):
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    individual[mask] = 1 - individual[mask]
    return individual

def evolve(pop, generations):
    best_fitness_per_gen = []

    for _ in range(generations):
        fits = np.array([fitness(ind) for ind in pop])
        gen_best_idx = np.argmax(fits)
        gen_best_f = fits[gen_best_idx]
        best_fitness_per_gen.append(float(gen_best_f))

        new_pop = []
        while len(new_pop) < len(pop):
            p1 = tournament_selection(pop, fits, TOURNAMENT_K)
            p2 = tournament_selection(pop, fits, TOURNAMENT_K)
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:len(pop)], dtype=np.int8)
    return pop, best_fitness_per_gen

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title="Bit Pattern GA", page_icon="🧬")
st.title("🧬 Genetic Algorithm: Evolve an 80-bit Pattern")
st.markdown("This app evolves a population of 80-bit chromosomes to reach a target of 50 ones.")

# Initialize population
pop = init_population(POP_SIZE, CHROM_LEN)

# Run GA
pop, best_fitness_per_gen = evolve(pop, N_GENERATIONS)

# Plotting the fitness curve
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(range(1, len(best_fitness_per_gen)+1), best_fitness_per_gen, marker='o', color='blue')
ax.set_xlabel("Generation")
ax.set_ylabel("Best Fitness")
ax.set_title("GA Best Fitness Over Generations")
ax.grid(True)
st.pyplot(fig)

# Display final result
best_individual = pop[np.argmax([fitness(ind) for ind in pop])]
best_fit = fitness(best_individual)
ones_count = best_individual.sum()

st.subheader("Final Result")
st.text(f"Best Individual: {best_individual}")
st.text(f"Number of Ones: {ones_count}")
st.text(f"Fitness: {best_fit}")

if best_fit == MAX_FITNESS and ones_count == TARGET_ONES:
    st.success(f"Perfect match achieved: ones = {ones_count} and fitness = {best_fit} ✅")
else:
    st.info(f"GA may reach near-optimal solutions: ones = {ones_count}, fitness = {best_fit}")
