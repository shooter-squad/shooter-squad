import os

import numpy as np
import cv2

from deep_q_network import DeepQNetwork
from grad_cam import GradCam
from observation_sampler import ObservationSampler
from Env import make_env

env_name = 'shooter'
env = make_env(env_name)

image_file_name = 'test'

def load_model():
    model = DeepQNetwork(0.001, env.action_space.n,
                         input_dims=env.observation_space.shape,
                         name=env_name + '_' + 'DQNAgentPretrain' + '_q_eval',
                         chkpt_dir='models/', pre_train_dir='dummy/')
    model.load_checkpoint()
    model.eval()

    return model


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name + '_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('./results', file_name + '_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    # org_img = cv2.resize(org_img, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


if __name__ == '__main__':
    observation_sampler = ObservationSampler(env)
    model = load_model()
    print(model)
    grad_cam = GradCam(model, target_layer='conv3')

    stacked_images = observation_sampler.sample_observation(np.random.randint(0, 100))
    cam = grad_cam.generate_cam(stacked_images)

    save_class_activation_on_image(stacked_images[0] * 255, cam, image_file_name)

    print(stacked_images[0].shape)
    cv2.imwrite('test.png', stacked_images[0] * 255)

