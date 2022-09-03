import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self,embed_size):
        super(EncoderCNN,self).__init__()
        self.inception = models.inception_v3(pretrained=True,aux_logits=False)
        # replacing last layer as an ordinary nn , in_features from cnn sent to last linear and embed_size as o/p number
        self.inception.fc = nn.Linear(self.inception.fc.in_features,embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,images):
        features = self.inception(images)
        return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers) :
        super(DecoderRNN,self).__init__()
        # embedding layer used to map or input word to certain dimnesion space to input it to the model
        # vocab size is index and its going to map it to embed_size(dimensionm )
        self.embed  = nn.Embedding(vocab_size , embed_size)
        # convert embed_size to hidden_size,num_layers is no. of lstm stacked upon each other
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self,features,captions):
        embeddings = self.dropout(self.embed(captions))        # converting to embeddings
        # features.unsqueeze(0) is aditional dimension which is added/concatinated to embedding so that its viewed as the time stamp
        # features are the extracted ouput for the Encoder CNN it's inputed as first word for the lstm to generate the o/p for real first word
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim=0)
        hiddens,_ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers) :
        super(CNNtoRNN,self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers)

    def forward(self,images,captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features,captions)
        return outputs

    def caption_image(self,image,vocabulary,max_length=50):
        result_caption = []

        # so that we can have dimension for the batch 
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None # hidden state for lstm

        for _ in range(max_length):
            # using .unsqueeze(0)  o convert image to the batch dimension

            hiddens , states = self.decoderRNN.lstm(x,states)
            output = self.decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)   # the word with highest probability
            # print(predicted)
            result_caption.append(predicted.item())

            # for next word taking predicting word as input  
            x = self.decoderRNN.embed(predicted).unsqueeze(0)  
            if vocabulary.itos[predicted.item()] == "<EOS>":   #itos is index to string
                break
        
        # idx - index here we convert index returned by lstm to string using our vocabulary
        return [vocabulary.itos[idx] for idx in result_caption]
