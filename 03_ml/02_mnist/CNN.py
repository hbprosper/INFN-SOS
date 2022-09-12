
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self):

        """
    A convolutional + fully-connected network for the MNIST
    problem.
        """ 
        # initialize base (or super, or parent) class
        super(Model, self).__init__()
        
        # LAYER 0
        # ----------------------------------------------------------
        # since we are using 5x5 filters, the addition of 2-pixel
        # padding around each image will allow cross-correlation of 
        # every pixel in the image, and every filter will be centered 
        # on each pixel as the filters are moved around the image. 
        # Since every pixel is processed, we obtain output images 
        # that are the same size as the input images.
        self.conv0 = nn.Conv2d(in_channels=1,    # input channels
                               out_channels=4,   # output channels
                               kernel_size=5,    # filter size (5x5)
                               stride=1,         # shift by this amount     
                               padding=2)        # pad by this amount
           
        # normalize each image; that is, normalize over all numbers
        # defined by the tensor indices (C=4, H=28, W=28).
        self.layernorm0= nn.LayerNorm(normalized_shape=(4, 28, 28))

        # down-sample with a (2,2) window (kernel_size), which shifts 
        #    horizontally or vertically with a stride of 2 pixels.
        #    This operation replaces the 4 channels of shape (28, 28) 
        #    with 4 channels of shape (14, 14) by replacing a group of 
        #    2x2 pixels in an input channel with a single pixel whose 
        #    value equals the largest pixel value among the 4 pixels 
        #    within the (2, 2) window.
        self.maxpool0  = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        # apply a relu non-linearity to every element of the tensor
        # input
        self.relu0     = nn.ReLU()
                
        # LAYER 1
        # ----------------------------------------------------------
        # instantiate a 2nd convolution layer
        # Note: the in_channels count must match the out_channels 
        # count of the previous layer.
        self.conv1 = nn.Conv2d(in_channels=4,    # input channels
                               out_channels=16,  # output channels
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.layernorm1 = nn.LayerNorm(normalized_shape=(16, 14, 14))
        self.maxpool1   = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.relu1      = nn.ReLU()
        
        # we end with a linear layer. Let's compute the number of
        # inputs to that layer. Ignoring the first index of the output
        # tensor from conv0 and conv1, which simply labels the ordinal 
        # value of the image in the batch of images, we note the 
        # following:
        # 1. conv0 outputs a tensor of size (4, 28, 28), that is, it
        #    outputs batches of 4 channels of shape (28, 28).
        #    we then down-sample to a tensor of size (4, 14, 14).
        #
        # 2. conv1 outputs a tensor of size (16, 14, 14), which will be
        #    down-sampled to one of size (16, 7, 7).
        #
        # 3. Therefore, when flattened, the number of inputs to the
        #    linear layer is: 16 * 7 * 7 = 784 (which, just happens to
        #    equal the number of pixels in the original images!)
        self.n_inputs = 16 * 7 * 7
        
        # 4. We have 10 outputs, one for each digit
        self.linear   = nn.Linear(self.n_inputs, 10)

        # see description in forward(...) method
        self.dropout  = nn.Dropout(p=0.5)

        # see description in forward(...) method
        self.softmax  = nn.Softmax(dim=1)
    
    # define (required) method to compute output of network
    def forward(self, x):
        # conv0 expects a 4-d tensor of shape 
        # (N=batch_size, C=channels, H=height, W=width). So we must 
        # reshape x. The -1 index means the batch size is to be
        # inferred at runtime from the tensor x.
        y = x.view(-1, 1, 28, 28)
        
        # LAYER 0
        # 1. cross-correlate the input tensor of shape (-1, 1, 28, 28),
        #    padded with a 2-pixel wide strip, with a (4, 1, 5, 5) 
        #    kernel, thereby producing an output tensor of shape 
        #    (-1, 4, 28, 28).
        y = self.conv0(y)
        
        # 2. normalize images.
        y = self.layernorm0(y)
        
        # 3. down-sample with a (2,2) window, which shifts horizontally
        #    or vertically 2 pixels at a time. This replaces the 4 
        #    channels of shape (28, 28) with 4 channels of shape (14, 14) 
        #    by replacing a group of 2x2 pixels in an input channel with 
        #    the largest pixel value among the 4 pixels. The output tensor 
        #    at this stage has shape (-1, 4, 14, 14).
        y = self.maxpool0(y)
        
        # 4. apply a relu non-linearity to every element of this tensor
        y = self.relu0(y)
        
        # LAYER 1
        # 1. cross-correlate a (-1, 4, 14, 14) tensor, padded as above, 
        #    with a (16, 4, 5, 5) kernel and, for each (5, 5) filter, 
        #    sum over the 4 input channels. Since the kernel contains 
        #    16 output channels, the end result is a 16-channel image. 
        #    The output, therefore, has shape (-1, 16, 14, 14).
        y = self.conv1(y) 
        
        # 2. normalize the 16-channel image of shape (14, 14)
        y = self.layernorm1(y)
        
        # 3. down-sample with a (2,2) window, as above, thereby 
        #    creating an output tensor of shape (-1, 16, 7, 7).
        y = self.maxpool1(y)
        
        # 4. apply a relu function element-wise (as above).
        y = self.relu1(y)
       
        # flatten the tensor (-1, 16, 7, 7) to the tensor (-1, 16*7*7).
        y = y.view(-1, self.n_inputs)
        
        # During training, randomly dropout, that is, zero, 
        # half of the elements in the current tensor y. Dropout has
        # been shown to reduce the tendency to overtrain.
        # Dropout effectively deactivates all the weights attached 
        # to the zeroed element. Alternatively, it can be thought of 
        # as a way to apply random modifications to a multi-channel 
        # image by randomly setting half the pixels to zero at each 
        # iteration.
        if self.training:
            y = self.dropout(y)
            
        # Apply a linear transformation to the (-1, 784) tensor.
        # We could use more than one linear layer here, which may
        # (or may not!) yield better results.
        y = self.linear(y)
        
        # Apply the softmax function horizontally, i.e., along 
        # the class axis (dim=1) in order to ensure that the outputs 
        # sum to unity.
        # (Note: dim=0 is vertical, that is, along the batch axis.)
        
        # Final output: estimated class probabilities for ith image,
        #   q_i(k) = exp(y_i(k) / sum_j exp(y_i(j)), j = 0,..., K-1,
        # where K=10 is the number of classes and y_i(k) is the output 
        # for the ith image for class index k. 
        y = self.softmax(y)

        return y
    
# Here is a much simpler way to implement the same model!

model = nn.Sequential(nn.Conv2d(in_channels=1,   # input channels
                                out_channels=4,  # output channels
                                kernel_size=5,   # 5x5 filter size
                                stride=1,
                                padding=2),
                      nn.LayerNorm(normalized_shape=(4, 28, 28)),
                      nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                      nn.ReLU(),
                      
                      nn.Conv2d(in_channels=4,   # input channels
                                out_channels=16, # output channels
                                kernel_size=5,
                                stride=1,
                                padding=2),
                      nn.LayerNorm(normalized_shape=(16, 14, 14)),
                      nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                      nn.ReLU(),
        
                      nn.Flatten(),
                      nn.Linear(784, 10),
                      nn.Dropout(p=0.5),
                      nn.Softmax(dim=1) 
        )
