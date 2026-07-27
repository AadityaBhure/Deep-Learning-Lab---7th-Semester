#PRN: 23070521034
#NAME: Aaditya Bhure
import numpy as np
import matplotlib.pyplot as plt

# Dataset
x = np.array([100, 150, 200, 250, 300])
y = np.array([20, 25, 30, 35, 40])

print("Input X:", x)
print("Input Y:", y)

# Mean
mean_x = np.mean(x)
mean_y = np.mean(y)

print("Mean of X:", mean_x)
print("Mean of Y:", mean_y)

# Calculate slope
numerator = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sum((x - mean_x) ** 2)

b1 = numerator / denominator
print("Slope (b1):", b1)

# Calculate intercept
b0 = mean_y - b1 * mean_x
print("Intercept (b0):", b0)

# Prediction
y_pred = b0 + b1 * x

print("Original Y:", y)
print("Predicted Y:", y_pred)

# Residuals
residuals = y - y_pred

print("Residuals:", residuals)

# Error Metrics
mae = np.mean(np.abs(residuals))
mse = np.mean(residuals ** 2)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
