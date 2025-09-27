import math

def calculate_y(x):
    numerator = math.log(math.exp(-x)) + math.cos(x - 1)
    denominator = (math.log(x) ** 3) + math.sqrt(math.sin(3 * x) + math.cos(2 * x))
    
    if denominator == 0:
        raise ValueError("Denominator is zero, division by zero is not allowed.")
    
    return numerator / denominator

x_value = 2.4
try:
    y_value = calculate_y(x_value)
    print(f"y = {y_value}")
except ValueError as e:
    print(e)