import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp

class ImageMask:
    """
    an image mask is a boolean matrix with shape (H, W, C)
    """
    def __init__(self, mask_array) -> None:
        self._mask = mask_array

    def visualise(self):
        for c_index in range(self._mask.shape[2]):
            plt.subplot(1,self._mask.shape[2],c_index+1)
            plt.imshow(self._mask[:,:,c_index], vmin=0, vmax=1)

class CheckboardMask(ImageMask):
    def __init__(self, shape) -> None:
        x, y = np.arange(shape[0], dtype=np.int32), np.arange(shape[1], dtype=np.int32)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        mask = np.fmod(xx + yy, 2)
        mask = mask.astype(bool).reshape(shape[0], shape[1], 1)
        mask = np.repeat(mask, shape[2], axis=2)

        super().__init__(mask)

class ChannelMask(ImageMask):
    def __init__(self, shape) -> None:
        m = np.zeros((*shape[0:2],1))
        masks = []
        for i in range(shape[2]):
            masks.append(m % 2)
            m = m + 1
        mask = np.stack(masks,axis=2)[:,:,:,0]
        super().__init__(mask)

if __name__ == '__main__':
    plt.figure()
    # CheckboardMask((28,28,1)).visualise()
    ChannelMask((28,28,1)).visualise()
    plt.show()

    plt.figure()
    # CheckboardMask((14,14,4)).visualise()
    ChannelMask((14,14,4)).visualise()
    plt.show()

    event_shape = (28,28,1)
    mask = np.arange(0, onp.prod(event_shape)) % 2
    mask = np.reshape(mask, event_shape)
    mask = mask.astype(bool)

    plt.figure()
    ImageMask(mask_array=mask).visualise()
    plt.show()


    event_shape = (28,28,1)
    mask = np.arange(0, np.prod(np.array(event_shape))) % 2
    mask = np.reshape(mask, event_shape)
    mask = mask.astype(bool)

    plt.figure()
    ImageMask(mask_array=mask).visualise()
    plt.show()
