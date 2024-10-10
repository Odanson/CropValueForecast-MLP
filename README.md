```markdown
# CropValueForecast-MLP

This repository contains a machine learning project aimed at forecasting the export value of crop products using a Multilayer Perceptron (MLP) model. The model uses historical agricultural data to predict crop export values three years into the future.

## Project Description

The goal of this project is to forecast the export value of crop products using a variety of food and agricultural indicators sourced from the FAOSTAT database. The model is built using Python and TensorFlow/Keras, and leverages a feedforward neural network for regression.

## Dataset

The dataset used in this project is derived from the FAOSTAT database and includes the following categories:

- Consumer prices indicators
- Crops production indicators
- Emissions
- Employment
- Exchange rate
- Fertilizers use
- Food balances indicators
- Food security indicators
- Food trade indicators
- Foreign direct investment
- Land temperature change
- Land use
- Pesticides use

The data covers over 245 countries and spans from the mid-1990s to the present day. It includes variables relevant to food and agriculture, such as crop yields, prices, emissions, and more.

## Data Processing

The preprocessing steps in this project include:

- **Scaling**: The features were scaled using `MinMaxScaler` to ensure all variables were on the same scale.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was used to reduce the dimensionality of the data while retaining most of the variance.
- **Transformations**: The `PowerTransformer` was applied to normalize data distributions and reduce skewness.

## Model

The model used for this project is a Multilayer Perceptron (MLP) implemented using TensorFlow/Keras. Key details of the model are as follows:

- **Architecture**: The model consists of an input layer, two hidden layers, and a single output layer.
- **Layers**:
  - Input Layer: Takes the preprocessed features.
  - Hidden Layers: Dense layers with ReLU activation functions and L2 regularization to prevent overfitting.
  - Output Layer: A single unit that predicts the export value (regression task).
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Regularization**: Dropout layers and L2 regularization were applied to prevent overfitting.
  
## Performance

The model's performance was evaluated using metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on the test set. The model was trained and validated on an 80/20 split of the data.

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/CropValueForecast-MLP.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook and run the model:
   ```bash
   jupyter notebook Final_Code_Candid_No_276221.ipynb
   ```
4. The model predictions and performance metrics will be displayed in the output cells.

## Dependencies

- Python 3.x
- TensorFlow/Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Results

The results of the model, including the predictions and performance metrics, can be found in the notebook outputs. The `predictions.csv` file includes the actual vs predicted export values.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
