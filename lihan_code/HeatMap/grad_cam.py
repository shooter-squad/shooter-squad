import numpy as np
import torch
import torch.nn.functional as F
import cv2


class CamExtractor:
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_name, module in self.model._modules.items():
            print(module_name)
            if module_name == 'fc1':
                return conv_output, x
            x = F.relu(module(x))  # Forward
            # print(module_name, module)
            if module_name == self.target_layer:
                print('True')
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = F.relu(self.model.fc1(x))
        x = self.model.fc2(x)
        return conv_output, x


class GradCam:
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model.cuda()
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_index=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)\
        input_image = torch.tensor(input_image).cuda().float()
        input_image = torch.unsqueeze(input_image, 0)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_index is None:
            target_index = np.argmax(model_output.cpu().data.numpy())
        # Target for backprop
        model_output.cuda()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_index] = 1
        # Zero grads
        self.model.fc1.zero_grad()
        self.model.fc2.zero_grad()
        # self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output.cuda(), retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
        # Get convolution outputs
        target = conv_output.cpu().data.numpy()[0]
        # Get weights from gradients
        # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (84, 84))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) -
                                     np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam
