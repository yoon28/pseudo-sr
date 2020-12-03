import cv2
import numpy as np

def tensor_to_image(tensor, img_range, rgb=True):
    m_tens = tensor.detach().squeeze().cpu()
    assert len(m_tens.size()) == 3
    arrays = np.clip(m_tens.numpy().transpose(1, 2, 0), a_min=0, a_max=img_range) / img_range
    img = np.around(arrays * 255).astype(np.uint8)
    if rgb: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def save_tensor_image(file_name, tensor, img_range, rgb=True):
    img = tensor_to_image(tensor, img_range=img_range, rgb=rgb)
    cv2.imwrite(file_name, img)

class AverageMeter():
    def __init__(self, data=None, ema_alpha=None):
        self.n_samples = 1 if data is not None else 0
        self.average = data if data is not None else 0
        self.ema_alpha = ema_alpha

    def update(self, data):
        if self.ema_alpha is None:
            self.average = (self.n_samples) / (self.n_samples + 1) * self.average + 1 / (self.n_samples + 1) * data
        else:
            self.average = (1 - self.ema_alpha) * self.average + self.ema_alpha * data if self.n_samples >= 1 else data
        self.n_samples += 1

    def get_avg(self):
        return self.average

    def reset(self):
        self.n_samples = 0
        self.average = 0