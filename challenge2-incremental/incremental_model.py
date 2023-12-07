"""
    Basic pytorch autoencoder, based on code from:

    https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/
"""
import torch
from torchvision import datasets
from torchvision import transforms
import pickle
import base64
import random

#set a seed for reproducibillity (deterministic algorithms)
def reset_seeds():
    random.seed(0)
    torch.manual_seed(0)

reset_seeds()
torch.use_deterministic_algorithms(True)

#if using numpy, you may have to set its seed as well
#import numpy as np
#np.random.seed(0)

#cuda gpu settings:
#torch.backends.cudnn.benchmark = False

def get_dataset(segment):
    # Transforms images to a PyTorch Tensor
    tensor_transform = transforms.ToTensor()

    # Download the MNIST Dataset
    dataset = datasets.MNIST(root = "./data",
                             train = True,
                             download = True,
                             transform = tensor_transform)
    evens = list(range(0, len(dataset), 2))
    odds = list(range(1, len(dataset), 2))

    if segment == 0:
        trainset = torch.utils.data.Subset(dataset, evens)
    else:
        trainset = torch.utils.data.Subset(dataset, odds)


    loader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                         shuffle=False, num_workers=2)
    return loader


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    
class AutoEncoderTrainer:
    epochs = 0
    segments_results = [{"updated": False, "losses": []}, 
                        {"updated": False, "losses": []}]
    model = AE()
    new_model = AE()
    updates = 0

    @classmethod
    def get_data(cls):
        serialized = base64.b64encode(pickle.dumps(cls.model.state_dict())).decode("utf-8")
        return serialized, cls.epochs

    @classmethod
    def post_data(cls, parameters_dict, segment, loss):
        if cls.segments_results[segment]['updated'] == True:
            return False

        if loss not in cls.segments_results[segment]['losses']:
            cls.segments_results[segment]['losses'].append(loss)
            return True

        #We have a confirmed reproduced loss for this segment, so update the model.
        cls.segments_results[segment]['updated'] = True

        state_dict = pickle.loads(base64.b64decode(parameters_dict.decode("utf-8")))
        model_dict = cls.new_model.state_dict()
        for key in model_dict:
            if cls.updates == 0:
                model_dict[key] = state_dict[key]/2;
            else:
                model_dict[key] = model_dict[key]+state_dict[key]/2;
        cls.new_model.load_state_dict(model_dict)

        if cls.updates == 1:
            cls.epochs += 1
            cls.updates = 0 
            cls.model = cls.new_model
            cls.new_model = AE()
            cls.segments_results = [{"updated": False, "losses": []}, 
                                    {"updated": False, "losses": []}]
        else:
            cls.updates+=1
        return True

    @classmethod
    def train(cls, model, epochs, segment):
        loader = get_dataset(segment)
        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()
 
        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
        for epoch in range(epochs):
            losses = []
            for (image, _) in loader:
              # Reshaping the image to (-1, 784)
              image = image.reshape(-1, 28*28)
       
              # Output of Autoencoder
              reconstructed = model(image)
       
              # Calculating the loss function
              loss = loss_function(reconstructed, image)
       
              # The gradients are set to zero,
              # the gradient is computed and stored.
              # .step() performs parameter update
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
       
              # Storing the losses in a list for plotting
              losses.append(loss)
            total_loss = sum(losses)/len(losses)
        return total_loss.tolist()


def main():
    while True:
        #client 1 here
        reset_seeds()
        model1 = AE()
        serialized_data, epochs = AutoEncoderTrainer.get_data()
        state_dict = pickle.loads(base64.b64decode(serialized_data.encode('utf-8')))
        model1.load_state_dict(state_dict)
        loss = AutoEncoderTrainer.train(model1, 1, 0)
        AutoEncoderTrainer.post_data(pickle.dumps(model1.state_dict()), 0, loss)
        print("Client: 1 Epochs:", epochs, "Loss:", loss, "Segment: 0")

        reset_seeds()
        model2 = AE()
        serialized_data, epochs = AutoEncoderTrainer.get_data()
        state_dict = pickle.loads(base64.b64decode(serialized_data.encode('utf-8')))

        model2.load_state_dict(state_dict)
        loss = AutoEncoderTrainer.train(model2, 1, 1)
        AutoEncoderTrainer.post_data(pickle.dumps(model2.state_dict()), 1, loss)
        print("Client: 2 Epochs:", epochs, "Loss:", loss, "Segment: 1")

        reset_seeds()
        model3 = AE()
        serialized_data, epochs = AutoEncoderTrainer.get_data()
        state_dict = pickle.loads(base64.b64decode(serialized_data.encode('utf-8')))
        model3.load_state_dict(state_dict)
        loss = AutoEncoderTrainer.train(model3, 1, 0)
        AutoEncoderTrainer.post_data(pickle.dumps(model3.state_dict()), 0, loss)
        print("Client: 3 Epochs:", epochs, "Loss:", loss, "Segment: 0")

        reset_seeds()
        model4 = AE()
        serialized_data, epochs = AutoEncoderTrainer.get_data()
        state_dict = pickle.loads(base64.b64decode(serialized_data.encode('utf-8')))
        model4.load_state_dict(state_dict)
        loss = AutoEncoderTrainer.train(model4, 1, 1)
        AutoEncoderTrainer.post_data(pickle.dumps(model4.state_dict()), 1, loss)
        print("Client: 4 Epochs:", epochs, "Loss:", loss, "Segment: 1")

    import pdb
    pdb.set_trace()


if __name__=='__main__':
    main()
