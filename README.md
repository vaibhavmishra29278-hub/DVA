Practical no 1 Implementation of single layer perceptron

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def fit(self, features, labels):
        self.weights = np.zeros(features.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            for x, y in zip(features, labels):
                prediction = int(np.dot(x, self.weights) + self.bias >= 0)
                update = self.learning_rate * (y - prediction)
                self.weights += update * x
                self.bias += update

    def predict(self, features):
        return (np.dot(features, self.weights) + self.bias >= 0).astype(int)

# Load dataset (2 classes only)
iris = load_iris()
features = iris.data[iris.target != 2]
labels = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = Perceptron()
model.fit(X_train, y_train)

accuracy = np.mean(model.predict(X_test) == y_test)
print("Accuracy:", accuracy)

# Decision boundary using first two features
train_2d = X_train[:, :2]
model.fit(train_2d, y_train)

x_grid, y_grid = np.meshgrid(
    np.linspace(train_2d[:,0].min(), train_2d[:,0].max(), 100),
    np.linspace(train_2d[:,1].min(), train_2d[:,1].max(), 100)
)

grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]
predictions = model.predict(grid_points).reshape(x_grid.shape)

plt.contourf(x_grid, y_grid, predictions, alpha=0.5)
plt.scatter(train_2d[:,0], train_2d[:,1], c=y_train, cmap="coolwarm")
plt.title("Perceptron Decision Boundary")
plt.show()

Practical no 2 Implementation of multi-layer perceptron

!pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Model
model = nn.Sequential(
    nn.Linear(3,8), nn.ReLU(),
    nn.Linear(8,16), nn.ReLU(),
    nn.Linear(16,2)
)

# Dummy data
X = torch.randn(100,3)
y = torch.randint(0,2,(100,))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []

# Training
for _ in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Loss plot
plt.plot(losses)
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.show()

# Accuracy
with torch.no_grad():
    pred = model(X).argmax(1)
    print("Accuracy:", (pred==y).float().mean().item())

Practical no 3 Implementation of Bidirectional Associative Memory (BAM)

import numpy as np

class BAM:
    def __init__(self, x_neurons, y_neurons):
        self.w = np.zeros((x_neurons, y_neurons))
    def train(self, patterns):
        for x, y in patterns:
            self.w += np.outer(x, y)
    def recall_x(self, x):
        return np.sign(x @ self.w)
    def recall_y(self, y):
        return np.sign(y @ self.w.T)


# Example
bam = BAM(4, 2)

patterns = [
    (np.array([1, 1, 1, -1]), np.array([1, 1])),
    (np.array([-1, -1, 1, 1]), np.array([-1, 1]))
]

bam.train(patterns)

print("Weights:\n", bam.w)

x = np.array([1, 1, 1, -1])
print("\nInput X:", x)
print("Recalled Y:", bam.recall_x(x))

y = np.array([-1, 1])
print("\nInput Y:", y)
print("Recalled X:", bam.recall_y(y))

noisy_x = np.array([1, -1, 1, -1])
print("\nNoisy X:", noisy_x)
print("Recalled Y:", bam.recall_x(noisy_x))

Practical no 4Implementation of fuzzy logic.

def fuzzy_logic(temp, humid):
    def tri(x, a, b, c):
        if x <= a or x >= c: return 0
        if x == b: return 1
        if x < b: return (x-a)/(b-a)
        return (c-x)/(c-b)
    # Temperature memberships
    t_low  = 1 if temp <= 25 else (50-temp)/25 if temp < 50 else 0
    t_med  = tri(temp, 25, 50, 75)
    t_high = 0 if temp <= 50 else (temp-50)/25 if temp < 75 else 1

    # Humidity memberships
    h_low  = 1 if humid <= 25 else (50-humid)/25 if humid < 50 else 0
    h_med  = tri(humid, 25, 50, 75)
    h_high = 0 if humid <= 50 else (humid-50)/25 if humid < 75 else 1

    # Defuzzification
    low  = (t_low + h_low) / 2
    med  = (t_med + h_med) / 2
    high = (t_high + h_high) / 2

    return [low, med, high]


temps  = [23, 45, 56, 78]
humids = [56, 45, 78, 78]

for t, h in zip(temps, humids):
    low, med, high = fuzzy_logic(t, h)
    print(f"For Temperature:{t} and Humidity:{h} \nFan Spped --> Low:{low:.2f}, Med:{med:.2f}, High:{high:.2f}\n")

Practical no 5 Implementation of heb rule learning

import numpy as np

def hebbian_learning(X, y, lr=0.1):
    w = np.zeros(X.shape[1])
    print("Initial weights:", w)

    for i, (x, target) in enumerate(zip(X, y), 1):
        dw = lr * x * target
        print(f"Delta weight: {dw}\n")
        w += dw
        print(f"After sample {i}: {w}")

    return w


X = np.array([[1,0],
              [0,1],
              [1,1],
              [0,0]])

y = np.array([0,0,1,0])

w = hebbian_learning(X, y)
print("\nFinal weights:", w)

Practical no 6 Implementation of self organizing map

import numpy as np

class SOM:
    def __init__(self, size, dim):
        self.size = size
        self.w = np.random.rand(size, size, dim)

    def train(self, data, lr=0.1, radius=2, epochs=100):
        for _ in range(epochs):
            for x in data:
                d = np.linalg.norm(self.w - x, axis=2)
                bmu = np.unravel_index(np.argmin(d), d.shape)

                for i in range(self.size):
                    for j in range(self.size):
                        dist = np.sqrt((i-bmu[0])**2 + (j-bmu[1])**2)
                        if dist <= radius:
                            self.w[i,j] += lr * (x - self.w[i,j])

data = np.random.rand(100,3)

som = SOM(5,3)
som.train(data)

print("Final Weights:\n", som.w)

Practical no 7 Implementation of delta rule learning

import numpy as np

def delta_rule_learning(inputs, targets, learning_rate=0.1, epochs=5):
    weights = np.zeros(inputs.shape[1])
    print("Initial Weights:", weights)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        for input_vector, target in zip(inputs, targets):

            output = np.dot(input_vector, weights)
            error = target - output

            weight_update = learning_rate * error * input_vector
            weights += weight_update

            print("Input:", input_vector,
                  "Output:", round(output, 2),
                  "Error:", round(error, 2),
                  "Weights:", weights)

    return weights


# Dataset
inputs = np.array([
    [1, 1],
    [0, 1],
    [1, 0],
    [0, 0]
])

targets = np.array([1, 0, 0, 0])

# Train model
final_weights = delta_rule_learning(inputs, targets)

print("\nFinal Weights:", final_weights)

Practical 8 Genetic Algorithm

import random

TARGET = "I love MSCDSAI with RJColleges"
GENES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890 "
POP_SIZE = 500

def random_string():
    return [random.choice(GENES) for _ in range(len(TARGET))]

def fitness(chromosome):
    return sum(c != t for c, t in zip(chromosome, TARGET))

def mate(p1, p2):
    child = []
    for g1, g2 in zip(p1, p2):
        r = random.random()
        if r < 0.45:
            child.append(g1)
        elif r < 0.90:
            child.append(g2)
        else:
            child.append(random.choice(GENES))  # mutation
    return child

# Initial population
population = [random_string() for _ in range(POP_SIZE)]
generation = 1

while True:
    population = sorted(population, key=fitness)

    if fitness(population[0]) == 0:
        break

    new_population = population[:POP_SIZE//10]  # best 10%

    while len(new_population) < POP_SIZE:
        p1 = random.choice(population[:POP_SIZE//2])
        p2 = random.choice(population[:POP_SIZE//2])
        new_population.append(mate(p1, p2))

    population = new_population

    print(f"Gen {generation}: {''.join(population[0])}")
    generation += 1

print("\nTarget Reached:", "".join(population[0]))

