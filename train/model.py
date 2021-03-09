import torch.nn as nn
import torch
class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, batch_size, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size#
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim#
        self.n_layers = n_layers#
        self.batch_size = batch_size#
        
        # define model layers
        self.embed = nn.Embedding(vocab_size,embedding_dim)#
        self.lstm= nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, 
                             batch_first =True)
        self.dropout = nn.Dropout(dropout)
        self.fc= nn.Linear(hidden_dim,output_size)
        self.sig = nn.Sigmoid()
        self.train_on_gpu= torch.cuda.is_available()
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function   
        self.batch_size = nn_input.size(0)#
        nn_input = self.embed(nn_input)#
        self.lstm.flatten_parameters()#
        lstm_output,hidden = self.lstm(nn_input,hidden)#
        #lstm_output = self.dropout(lstm_output)#
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)
        #output = self.sig(self.fc(lstm_output))
        output = self.fc(lstm_output)
        # reshape into (batch_size, seq_length, output_size)
        output = output.view(self.batch_size, -1, self.output_size)
        # get last batch
        out = output[:, -1]
        # return one batch of output word scores and the hidden state
        return out, hidden
    
    
    def init_hidden(self):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        weight = next(self.parameters()).data
        batch_size = self.batch_size
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        
        # initialize hidden state with zero weights, and move to GPU if available
        
        return hidden
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