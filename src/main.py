# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the house-prices data
data = pd.read_csv('../data/housing.csv')
# print(data.head())

#Data setup
x = data['Size'].values
y = data['Price'].values

# Normalizing x
x_mean = x.mean()
x_std = x.std()
x_norm = (x - x_mean) / x_std

#Visualize the data
plt.xticks([500,1000,1500,2000,2500])
plt.yticks([150,300,450,600,750])

plt.title('House Price Prediction',fontfamily='Serif',fontsize='25',color='magenta')
plt.grid(True, linestyle='--', alpha=0.3)

plt.xlabel('Size (1000 sq.ft)', fontfamily='roboto', fontsize='20', color='seagreen')
plt.ylabel('Price ($1000s)', fontfamily='mono space', fontsize='25', color='seagreen')

plt.scatter(x,y, color='blue', marker='x')
plt.tight_layout()
plt.plot()
plt.show()

#compute cost function
def compute_cost(x, y, w, b):
    m =x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w *x[i] + b
        cost += (f_wb-y[i]) **2
    cost = cost/(2*m)
    return cost

cost= compute_cost(x, y, w=0, b=0)
print(f"Total cost is {cost}")             # Total cost is 123750.0

# compute gradient descent
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_db = 0
    dj_dw = 0
    for i in range(m):
        f_wb = w*x[i] +b
        error = f_wb - y[i]
        dj_dw += error*x[i]
        dj_db += error 
    dj_dw/=m
    dj_db/=m
    return dj_dw,dj_db              #     dj_dw and dj_db is -825000 & -450

dj_dw,dj_db= compute_gradient(x,y,w=0,b=0)
print(f"The value of dj_dw and dj_db is {dj_dw},{dj_db}")

#Gradient Descent
def gradient_descent(x ,y ,w_init ,b_init ,alpha ,num_iters):
    w = w_init
    b = b_init

    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient(x, y, w, b)
        w -= alpha*dj_dw
        b -= alpha*dj_db

        if i % 100 == 0:
            cost=compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost={cost:.2f}, w={w:.2f}, b={b:.2f}")

    return w, b                       # w: 212.12, b: 449.98

#Train the Model
w_init = 0
b_init = 0
alpha = 0.01
num_iters = 1000

# Train the model using normalized x
w_final,b_final = gradient_descent(x_norm, y, w_init, b_init, alpha, num_iters)
print(f'Trained w: {w_final:.2f}, b: {b_final:.2f}')

# Plot prediction line (with original x)
x_range = np.linspace(min(x), max(x), 100)
x_range_norm = (x_range - x_mean) / x_std
y_pred = w_final * x_range_norm + b_final

plt.figure(figsize=(8,5))
plt.title('Trained Linear Regression Model', fontsize=20, fontfamily='serif')
plt.xlabel('Size (sq.ft)', fontsize=15)
plt.ylabel('Price ($1000s)', fontsize=15)
plt.scatter(x, y, color='blue', marker='x', label='Training Data')
plt.plot(x_range, y_pred, color='red', label='Prediction Line')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Predict Price
def predict(x, w, b, mean, std):
    x_norm = (x - mean) / std
    return w * x_norm + b

size = 1200  # in sq.ft
price = predict(size, w_final, b_final, x_mean, x_std)
print(f'Predicted price for {size} sq.ft house is ${price:.2f}k')

