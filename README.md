
# Linear Regression Visualization with Interactive Slider

This project demonstrates a simple linear regression model for predicting house prices based on square footage.
It includes an interactive slider to adjust the weight of the model and visualize the prediction line in real-time.

## Features
- Visualize the data points and the regression line dynamically.
- Adjust the weight of the regression line using an interactive slider.
- Real-time updates to the cost function and line based on the gradient descent optimization.

## Requirements
This script requires the following Python packages:
- `matplotlib`
- `numpy`
- `pandas`

You can install these packages via pip:
```bash
pip install matplotlib numpy pandas
```

## Usage
1. Place the `data_for_lr.csv` file in the `./data/` directory.
2. Run the script with:
   ```bash
   python script_name.py
   ```

3. Adjust the weight using the slider to see the effect on the regression line.

## Code Explanation
- `LinearRegression` class: This class represents a linear regression model and contains methods for prediction, plotting, cost calculation, and updating weights and biases.
- `update_weights_manually` function: This function updates the weight based on the slider input and adjusts the prediction line accordingly.
- Main script:
    - Initializes the data and linear regression model.
    - Trains the model for 100 iterations, with real-time visualization of the line updates.
    - Displays an interactive slider for manual adjustment of weights.

## Note
Ensure that the CSV file (`data_for_lr.csv`) contains two columns: `x` (square footage) and `y` (price).

## License
This project is open-source and available under the MIT License.
