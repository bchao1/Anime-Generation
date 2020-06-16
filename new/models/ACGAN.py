import torch
import torch.nn as nn

    
class Generator(nn.Module):
    """ ACGAN generator.
    
    ACGAN generator is simply a DCGAN generator that takes a noise vector and 
    class vector concatenated as input. All other details (activation functions,
    batch norm) follow the 2016 DCGAN paper.
    Attributes:
        latent_dim: the length of the noise vector
        class_dim: the length of the class vector (in one-hot form)
        gen: the main generator structure
    """


    def __init__(self, latent_dim, class_dim):
        """ Initializes Generator Class with latent_dim and class_dim."""
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.class_dim = class_dim

        self.gen = nn.Sequential(
                    nn.ConvTranspose2d(in_channels = self.latent_dim + 
                    								 self.class_dim, 
                                       out_channels = 1024, 
                                       kernel_size = 4,
                                       stride = 1,
                                       bias = False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 1024,
                                       out_channels = 512,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 512,
                                       out_channels = 256,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 256,
                                       out_channels = 128,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 128,
                                       out_channels = 3,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1),
                    nn.Tanh()
                    )
        return
    
    def forward(self, _input, _class):
        """ Defines the forward pass of the Generator Class.
        Args:
            _input: the input noise vector.
            _class: the input class vector. The vector need not be one-hot 
                    since multilabel generation is supported.
        
        Returns:
            The generator output.
        """


        concat = torch.cat((_input, _class), dim = 1)  # Concatenate noise and class vector.
        concat = concat.unsqueeze(2).unsqueeze(3)  # Reshape the latent vector into a feature map.
        return self.gen(concat)

class Discriminator(nn.Module):
    """ ACGAN discriminator.
    
    A modified version of the DCGAN discriminator. Aside from a discriminator
    output, DCGAN discriminator also classifies the class of the input image 
    using a fully-connected layer.

    Attributes:
    	num_classes: number of classes the discriminator needs to classify.
    	conv_layers: all convolutional layers before the last DCGAN layer. 
    				 This can be viewed as an feature extractor.
    	discriminator_layer: last layer of DCGAN. Outputs a single scalar.
    	bottleneck: Layer before classifier_layer.
    	classifier_layer: fully conneceted layer for multilabel classifiction.
			
    """
    def __init__(self, num_classes):
        """ Initialize Discriminator Class with num_classes."""
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(in_channels = 3, 
                             out_channels = 128, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 256, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 512, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace = True)
                    )   
        self.discriminator_layer = nn.Sequential(
                    nn.Conv2d(in_channels = 1024, 
                             out_channels = 1, 
                             kernel_size = 4,
                             stride = 1),
                    nn.Sigmoid()
                    ) 
        self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels = 1024, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2)
                    )
        self.classifier_layer = nn.Sequential(
                    nn.Linear(512, self.num_classes),
                    nn.Sigmoid()
                    )

        return
    
    def forward(self, _input):
        """ Defines a forward pass of a discriminator.
        Args:
            _input: A batch of image tensors. Shape: N * 3 * 64 *64
        
        Returns:
            discrim_output: Value between 0-1 indicating real or fake. Shape: N * 1
            aux_output: Class scores for each class. Shape: N * num_classes
        """

        features = self.conv_layers(_input)  
        discrim_output = self.discriminator_layer(features).view(-1) # Single-value scalar
        flatten = self.bottleneck(features).squeeze()
        aux_output = self.classifier_layer(flatten) # Outputs probability for each class label
        return discrim_output, aux_output

if __name__ == '__main__':
    latent_dim = 100
    class_dim = 31
    batch_size = 5
    z = torch.randn(batch_size, latent_dim)
    c = torch.randn(batch_size, class_dim)
    
    G = Generator(latent_dim, class_dim)
    D = Discriminator(class_dim)
    o = G(z, c)
    print(o.shape)
    x, y = D(o)
    print(x.shape, y.shape)