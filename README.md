Q1 35/94
for r in range(36):
    c = 35 - r
    if 4*r + 2*c == 94:
        print("Rabbits:", r)
        print("Chickens:", c)

Q2 C--B--C
import string

n = int(input("Enter a number: "))

alpha = string.ascii_lowercase

for i in range(n):
    s = "-".join(alpha[i:n])
    print((s[::-1] + s[1:]).center(4*n-3, "-"))

for i in range(n-2, -1, -1):
    s = "-".join(alpha[i:n])
    print((s[::-1] + s[1:]).center(4*n-3, "-"))

Q3 1+1 FOR 100
import time

start = time.time()

for i in range(100):
    x = 1 + 1

end = time.time()

print("Time:", end - start)

Q4 NORMALIZE 5X5 RANDOM 
import numpy as np

A = np.random.random((5,5))
A = (A - A.min())/(A.max() - A.min())

print(A)

Q5 4 DIFFERENT METHODS
import numpy as np

A = np.random.random(5)*10

print(np.floor(A))
print(A.astype(int))
print(np.trunc(A))
print([int(i) for i in A])

Q6 GENERATE 10 INTEGER 
import numpy as np

def gen():
    for i in range(10):
        yield i

A = np.array(list(gen()))
print(A)

Q7 ARRAY A AND B
import numpy as np

A = np.random.randint(0,5,5)
B = np.random.randint(0,5,5)

print(np.array_equal(A,B))

Q8 10X2 MATRIX
import numpy as np

A = np.random.random((10,2))

x = A[:,0]
y = A[:,1]

r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y,x)

print(r,theta)

Q9 100 RANDOM NUMBERS REPRESTING EXAM SCORES
import numpy as np
import matplotlib.pyplot as plt

scores = np.random.randint(0,100,100)

plt.hist(scores)
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Distribution of Scores")
plt.show()

Q10 LINE CHART BAR CHART
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5)
y = np.random.randint(1,10,5)

plt.figure()

plt.subplot(2,2,1)
plt.plot(x,y)

plt.subplot(2,2,2)
plt.bar(x,y)

plt.subplot(2,2,3)
plt.hist(np.random.randn(100))

plt.subplot(2,2,4)
plt.scatter(x,y)

plt.show()

Q11 VISUALIZATION DASHBOARD USING MATPLOTLIB THAT INCLUDES
import matplotlib.pyplot as plt

sales = [10,20,30,40]
profit = [5,10,15,20]
products = [40,30,20,10]

plt.figure()

plt.subplot(2,2,1)
plt.plot(sales)

plt.subplot(2,2,2)
plt.bar(range(4),profit)

plt.subplot(2,2,3)
plt.pie(products)

plt.show()

Q12 200 RANDOM EXAM SCORES BETWEEN O AND 100
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

scores = np.random.randint(0,100,200)

df = pd.DataFrame(scores, columns=["Marks"])

sns.histplot(df["Marks"])
plt.show()

Q13 THREE DIFFERENT CLASSES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "ClassA": np.random.randint(0,100,30),
    "ClassB": np.random.randint(0,100,30),
    "ClassC": np.random.randint(0,100,30)
}

df = pd.DataFrame(data)

sns.boxplot(data=df)
plt.show()

Q14 100 STUDENT WITH MULTIPLE ATTRIBUTES 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.DataFrame({
    "Marks": np.random.randint(0,100,100),
    "Study": np.random.randint(1,10,100),
    "Sleep": np.random.randint(4,10,100)
})

print(data.describe())

sns.histplot(data["Marks"])
plt.show()

sns.boxplot(data["Marks"])
plt.show()

sns.pairplot(data)
plt.show()

Q15 200 RANDOM VALUES REPRESENTING EXAM SCORES MEAN AND STANDARD DEVIATION
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

scores = np.random.randint(0,100,200)

print("Mean:", np.mean(scores))
print("Std Dev:", np.std(scores))

sns.histplot(scores)
plt.show()
