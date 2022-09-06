# Image-Captioning
A Decoder and encoder based seqtoseq model trained on lstm and inception_v3 to caption an Image

# Contents

[***Objective***](https://github.com/lollinng/Image-Captioning#objective)

[***Concepts***](https://github.com/lollinng/Image-Captioning#concepts)

[***Overview***](https://github.com/lollinng/Image-Captioning#overview)

[***Usage***](https://github.com/lollinng/Image-Captioning#Usage)



# Objective
**Building a model which outputs descriptive sentence for image we input**

We will be implementing the [_Show and Tell_](https://arxiv.org/abs/1411.4555) paper. This is by no means the current state-of-the-art, but it gave a breakthrough and an
idea to develop further paper like [_Show, Attend, and Tell_](https://arxiv.org/abs/1502.03044) and other transformer based implementation of the Image Captioning.

The model converts the pixels value of the image into feature vector outputed by a pretained cnn
which later gets inputed to the lstm network one time at a time to create an english sentence

Here are some ouputs of the model on the images not trained on 

![Screenshot (329)](https://user-images.githubusercontent.com/55660103/188632716-0c9cf495-d40d-49b1-8ee5-125ad08820a0.png)
p.s. the outputs on the left side are of left image and on the right side are of right image.

---

# Concepts

* **Transfer Learning**. We will be using a pretrained [inception_v3 model](https://pytorch.org/hub/pytorch_vision_inception_v3/) . It's convolutional neural network model that is 48 layers deep which was trained on more than a million images from the ImageNet database. We download the pretrained models so that we can recognize the features(different objects/contexts) from image pixels we provide as an input.

* **Encoder-Decoder architecture**. This type of architecture was first introduced in paper [_Seq to Seq_](https://arxiv.org/abs/1409.3215) . This arrangement of blocks helps us to get context vector from input and convert context vector into a sequence of output depending upon your business logic
In this example inceptionv3 is intialised in encoder which gives feature context vectore to the decoder which is LSTM in this case

* **LSTM**. LSTM or Long Short Term Memory is a type of rnn which is used to generate sequence of outputs by taking either sequence input or a vector value . [_LSTM_](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) was first introduced in 1997 and is still considered on of the best model for sequence data.


# Overview

In this section, I will present an overview of this model. If you're already familiar with it, you can skip straight to the [Implementation](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#implementation) section or the commented code.

### Encoder

The  Encoder gets feed images with a `batch_size=32 and img.shape of = (3*224*224)` This image is inputed to inception (whose last layer is replaced by our nueral layer) . 

The ouput features of the inception are then run through relu and dropout layers to increase dimnesionality and reduce overfiting respectively. 

![image_Cap](https://user-images.githubusercontent.com/55660103/188637846-a044d06d-2eee-4dff-95ca-ceeddce7848f.png)

### Decoder

The Decoder's task is to **Convert the context feature vector from encoder to a english sentence**.

We will be using and training an lstm in the decoder which will `take previous state ouput of iteself and feature vector as an input` and give us a word embedding as an 
ouput which when converted using our own vocabulary will provide us the predicted output sentence in english.

# Usage 

#### 1. Clone the repositories

```bash
git clone https://github.com/lollinng/Image-Captioning.git
cd Image-Captioning
```

#### 2. Download the dataset from and extract the files into flickr8k folder in project directory

```bash
https://www.kaggle.com/datasets/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb
```

#### 3. Hyperparamters

```bash
#check for hyper parameters at train.py
# check save_model and load_model paramters if u want to save/run a checkpint
```

#### 4. Train the model

```bash
python train.py    
```

#### 5. Test the model 

```bash
python run_model.py 
```



