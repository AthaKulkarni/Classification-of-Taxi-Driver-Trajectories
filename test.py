# test_model.py

import torch
from torch.utils.data import DataLoader, Dataset
from model import TaxiDriverClassifier  #, TaxiDriverClassifier2
from extract_feature import load_data, preprocess_data

from train import TaxiDriverDataset
import numpy as np
# torch.seed()

# Assuming the extract_feature.py handles preprocessing in a generalized way that can be applied to any data
## Using np.load to save time taken to load the data and preprocess it. Please load the data as per your method over here.
X_combined = np.load(r"C:\Users\Atharva Kulkarni\Desktop\WPI_Courses\Semester-2\Big Data Analytics\Atharva_project_2\X_combined.npy")
y_combined = np.load(r"C:\Users\Atharva Kulkarni\Desktop\WPI_Courses\Semester-2\Big Data Analytics\Atharva_project_2\y_combined.npy")

#Use the below code for importing the data from your data destination folder
# X_combined, y_combined = load_data(r"C:\Users\Atharva Kulkarni\Desktop\WPI_Courses\Semester-2\Big Data Analytics\Atharva_project_2\data_5drivers\*.csv")

        
def test(model, test_loader, device):
    """
    Test the model performance on the test set.
    """
    model.eval()
    test_loss = 0
    test_correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    ###########################
    # YOUR IMPLEMENTATION HERE #
    with torch.no_grad():
        for sample in test_loader:
            inputs, labels = sample["X"], sample["y"] 
            inputs, labels = inputs.to(device), labels.to(device)
            # import ipdb; ipdb.set_trace()
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, 0)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)

    ###########################

    return test_loss, test_correct


def test_model():
    """
    Main function to initiate the model testing process.
    Includes loading test data, setting up the model and test loader,
    and executing the testing function.
    """
    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ###########################
    # YOUR IMPLEMENTATION HERE #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # import ipdb; ipdb.set_trace()

    # Load test data
    test_loader = TaxiDriverDataset(X_combined, y_combined, device).test_loader
    # test_dataset = TaxiDriverDataset(test_data)#, transform=preprocess_data)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Set up the model
    # model = TaxiDriverClassifier()  # Initialize your model here
    input_dim, hidden_dim, output_dim = 8, 32, 5
    model = TaxiDriverClassifier(input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.2)
    model.load_state_dict(torch.load('logs/trial_logs9/model/best_acc_model.pt'))
    model.to(device)

    # Execute the testing function
    test_loss, test_correct = test(model, test_loader, device)

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_correct / len(test_loader.dataset) * 100:.2f}%")
    ###########################

if __name__ == '__main__':
    test_model()
