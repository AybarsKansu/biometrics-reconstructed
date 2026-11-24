"""Candidate generators: inputs are P vectors, settings, and a paired seed only.

The value received from yield is the visible score/code, never an identity,
hidden cosine score, recovery result, or held-out embedding.
"""
import numpy as np

from common import normalize


def zero_effort(pool, order):
    for position in order:
        yield pool[position]


def hill(pool, order, rng, settings):
    cursor = 0
    while True:
        current = pool[order[cursor % len(order)]].copy()
        cursor += 1
        best = yield current
        stalled = 0
        while stalled < settings["restart_after"]:
            direction = normalize(rng.normal(size=512))
            candidate = normalize(current + settings["step"] * direction)
            value = yield candidate
            if value > best:
                current, best, stalled = candidate, value, 0
            else:
                stalled += 1  # Equal codes keep the incumbent.


def simplex(pool, order, rng, settings):
    del rng  # Initial vertices and restart order already come from the paired seed.
    cursor = 0
    count = settings["vertices"]
    if count < 2 or len(order) < count:
        raise ValueError("Simplex needs at least two vertices and enough P embeddings.")
    while True:
        vertices, values = [], []
        for _ in range(count):
            vertex = pool[order[cursor % len(order)]].copy()
            cursor += 1
            value = yield vertex
            vertices.append(vertex)
            values.append(value)
        stalled = 0
        while stalled < settings["restart_after"]:
            ranking = sorted(range(count), key=lambda i: -values[i])
            vertices = [vertices[i] for i in ranking]
            values = [values[i] for i in ranking]
            previous_best = values[0]
            centroid = np.mean(vertices[:-1], axis=0)
            worst = vertices[-1]
            reflected = normalize(centroid + settings["reflection"] * (centroid - worst))
            reflected_value = yield reflected
            if reflected_value > values[0]:
                expanded = normalize(centroid + settings["expansion"] * (reflected - centroid))
                expanded_value = yield expanded
                if expanded_value > reflected_value:
                    vertices[-1], values[-1] = expanded, expanded_value
                else:
                    vertices[-1], values[-1] = reflected, reflected_value
            elif reflected_value > values[-2]:
                vertices[-1], values[-1] = reflected, reflected_value
            else:
                outside = reflected_value > values[-1]
                target = reflected if outside else worst
                contracted = normalize(centroid + settings["contraction"] * (target - centroid))
                contracted_value = yield contracted
                required = reflected_value if outside else values[-1]
                if contracted_value > required:
                    vertices[-1], values[-1] = contracted, contracted_value
                else:
                    for i in range(1, count):
                        vertices[i] = normalize(vertices[0] + settings["shrink"] * (vertices[i] - vertices[0]))
                        values[i] = yield vertices[i]
            stalled = 0 if max(values) > previous_best else stalled + 1


def spherical_crossover(first, second, fraction):
    cosine = float(np.clip(np.dot(first, second), -1, 1))
    if cosine < -0.9995:
        return first.copy()  # Antipodal parents have no unique short arc.
    if cosine > 0.9995:
        return normalize((1 - fraction) * first + fraction * second)
    angle = np.arccos(cosine)
    vector = (np.sin((1 - fraction) * angle) * first + np.sin(fraction * angle) * second) / np.sin(angle)
    return normalize(vector)


def genetic(pool, order, rng, settings):
    size = settings["population"]
    elite = settings["elite"]
    parents = max(2, int(size * settings["parent_fraction"]))
    if not 0 < elite < size or not 2 <= parents <= size or len(order) < size:
        raise ValueError("Invalid genetic population/elite/parent counts or too few P embeddings.")
    cursor = 0
    while True:
        population, values = [], []
        for _ in range(size):
            candidate = pool[order[cursor % len(order)]].copy()
            cursor += 1
            value = yield candidate
            population.append(candidate)
            values.append(value)
        stalled = 0
        while stalled < settings["restart_after"]:
            ranking = sorted(range(size), key=lambda i: -values[i])
            population = [population[i] for i in ranking]
            values = [values[i] for i in ranking]
            best = values[0]
            next_population = population[:elite]
            next_values = values[:elite]
            while len(next_population) < size:
                a, b = rng.choice(parents, 2, replace=False)
                child = spherical_crossover(population[a], population[b], rng.random())
                child = normalize(child + settings["mutation_sigma"] * rng.normal(size=512))
                value = yield child
                next_population.append(child)
                next_values.append(value)
            population, values = next_population, next_values
            stalled = 0 if max(values) > best else stalled + 1


def candidates(method, pool, seed, config):
    if len(pool) == 0:
        raise ValueError("The P embedding pool is empty.")
    if method in ("hill", "simplex", "genetic"):
        expected_ties = {"hill": "keep_incumbent", "simplex": "stable_keep_incumbent", "genetic": "stable_keep_earlier"}
        if config[method]["ties"] != expected_ties[method] or config[method]["restart_after"] < 1:
            raise ValueError("Unsupported tie policy or nonpositive restart interval.")
    order = np.random.default_rng(seed).permutation(len(pool))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    if method == "zero":
        return zero_effort(pool, order)
    if method == "hill":
        return hill(pool, order, rng, config["hill"])
    if method == "simplex":
        return simplex(pool, order, rng, config["simplex"])
    if method == "genetic":
        return genetic(pool, order, rng, config["genetic"])
    raise ValueError("Decision-based/1-bit attack experiments remain deferred.")
