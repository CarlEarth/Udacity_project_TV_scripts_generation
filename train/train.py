import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np

from model import RNN

def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    train_on_gpu= torch.cuda.is_available()
    # move data to GPU, if available
    if(train_on_gpu):
            inp, target = inp.cuda(), target.cuda()
    # perform backpropagation and optimization
    hidden = tuple([each.data for each in hidden])
    # zero accumulated gradients
    rnn.zero_grad()
    # get the output from the model
    output, hidden = rnn(inp, hidden)

    # calculate the loss and perform backprop
    loss = criterion(output.squeeze(), target.long())
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    #nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()
    #loss.item() calculate the average loss
    #do it
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")
    #Using the pd.read_csv output the dataframe 
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    train_y = torch.from_numpy(train_data[0].values).squeeze()
    #Squeeze:Returns a tensor with all the dimensions of input of size 1 removed.
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()
    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device,):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    # TODO: Paste the train() method developed in the notebook here.
    batch_losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        #model.train()
        total_loss = 0
        hidden = model.init_hidden()
        for batch_i,batch in enumerate(train_loader,1):         
            batch_X, batch_y = batch           
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)         
            #modify + training process 
            # forward, back prop
            loss, hidden = forward_back_prop(model, optimizer, loss_fn, batch_X, batch_y, hidden)          
            # record loss
            batch_losses.append(loss)
            show_every_n_batches=500
            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch, epochs, np.average(batch_losses)))
                batch_losses = []
        #print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))




if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=128, metavar='N',
                        help='size of the word embeddings (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')
    ##add modified part
    parser.add_argument('--output_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')
    parser.add_argument('--n_layers', type=int, default=2, metavar='N',
                        help='the number of layers/cells in your RNN (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='N',
                        help='the value of learning_rate (default: 0.001)')
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    #model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)
    #modified part
    #print("code_batch=",train_loader.size(0))
    print("setting_batch=",args.batch_size)
    model = RNN(args.vocab_size, args.output_size, args.embedding_dim, args.hidden_dim, args.n_layers, args.batch_size).to(device)
    #modified comment out
    #with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
    #    model.word_dict = pickle.load(f)

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}, output_size {}, n_layers {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size, args.output_size, args.n_layers
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
            'output_size': args.output_size, 
            'n_layers': args.n_layers
        }
        torch.save(model_info, f)

#	# Save the word_dict
#    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
#    with open(word_dict_path, 'wb') as f:
#        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
