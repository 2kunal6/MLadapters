import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils
from workflow.NN_workflow import NN_workflow


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test():
    """ Testing user defined model """

    # Step 1: Initialise the model.
    user_model = Net()

    # Step 2: Initialise the criterion.
    criterion = nn.CrossEntropyLoss()

    # Step 3: Initialise the NN_workflow based on the model and criterion.
    neural_net = NN_workflow(criterion, model=user_model)

    # Step 4: Initialise the optimizer.
    optimizer = torch.optim.SGD(user_model.parameters(), lr=0.001)

    # Step 5: Set the optimizer in the NN workflow.
    neural_net.set_optimizer(optimizer)

    # Step 6: Data preparation based on the dataset.
    train_loader, test_loader = neural_net.data_preparation(dset_name="MNIST")

    # Step 7: Call the train method of the NN workflow.
    neural_net.train(train_loader)

    # Step 8: Call the test method of the NN workflow.
    neural_net.test(test_loader)


if __name__ == "__main__":
    test()
