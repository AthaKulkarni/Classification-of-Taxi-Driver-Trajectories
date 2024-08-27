# Taxi Driver Trajectory Classification

This project uses deep learning to classify sequences, aiming to create a neural network model capable of predicting the driver associated with a given trajectory. The dataset includes daily driving trajectories for five taxi drivers over a six-month period. Each trajectory is defined by latitude and longitude coordinates, along with other relevant properties like date, time, and occupancy status.

## Project Overview

The project leverages the latitude and longitude of the trajectories to classify all GPS records from a single day for a specific driver. The dataset includes:

- **Latitude and Longitude:** These values indicate the position of the taxi.
- **Date and Time:** Complete date and time information for each trajectory.
- **Status Column:** Indicates whether the taxi is occupied (1) or not (0).

The model uses this information to predict the taxi plate number.

## Methodology

### Data Preprocessing

The data preprocessing script (`extract_feature.py`) loads and processes data from CSV files related to vehicle tracking. Key steps include:

1. **Data Loading:** The script uses the `glob` module to locate and process CSV files.
2. **Feature Extraction:** The timestamp is parsed to extract day, month, year, and time components (hour, minute, and second). 
3. **Scaling:** Features such as longitude, latitude, status, day, month, and time are scaled using `scikit-learn's StandardScaler`.
4. **Trajectory Formation:** Data is divided into 100-step chunks for each taxi plate.

### Network Structure

The project utilizes a Long Short-Term Memory (LSTM) neural network to predict taxi plate numbers. The network consists of:

- **Input Layer:** Receives the input sequence, where each vector represents a timestamp.
- **LSTM Layers:** Multiple LSTM layers process the sequence and generate hidden states.
- **Output Layer:** A linear layer converts the final hidden state into the output, predicting one of the five possible taxi plates.

### Model Training and Validation

The training process includes the following steps:

1. **Data Splitting:** Data is divided into training (80%) and testing (20%) sets using `sklearn.model_selection.train_test_split`.
2. **Scaling:** Features are scaled using `StandardScaler`.
3. **Model Definition:** The model is defined in the `TaxiDriverClassifier` class.
4. **Training Loop:** The model is trained for a specified number of epochs (125), with the `Adam` optimizer updating parameters.
5. **Validation:** After each epoch, the model is evaluated on the validation set, and performance metrics (loss and accuracy) are logged.

## Results

The project achieved a maximum validation accuracy of 86.86%. Training and validation results were logged, showing the loss and accuracy values per epoch. 

### Performance Comparison

Various models were experimented with during the project, including:

- **Linear Model:** Achieved up to 40% accuracy but was not suitable for sequential data.
- **LSTM Model:** The final LSTM model outperformed others after hyperparameter tuning.

### Hyperparameters

Key hyperparameters used in the project include:

- `hidden_size`: Number of neurons in the LSTM layer.
- `output_size`: Number of output neurons.
- `num_layers`: Number of LSTM layers.
- `learn_rate`: Learning rate for the optimizer.
- `num_epochs`: Number of training epochs.
- `batch_size`: Number of samples processed before updating weights.

## Conclusion

This project demonstrated the efficacy of LSTM neural networks for time series prediction tasks. The PyTorch framework was used to build and train the model, resulting in a maximum validation accuracy of 86.86%. The project highlighted the importance of data preprocessing and feature generation, as well as the selection of appropriate deep learning models for specific tasks.

## How to Run

1. Clone the repository.
2. Install the required libraries: `pandas`, `numpy`, `scikit-learn`, `PyTorch`, and `glob`.
3. Run the `extract_feature.py` script to preprocess the data.
4. Train the model using the `train.py` script.
5. Evaluate the model's performance on the validation set.

This project provides valuable insights into deep learning techniques for time series prediction, with potential applications in transportation-related scenarios.
