# Copyright (c) OpenMMLab. All rights reserved.
"""GaussianBlur transform for MMAction2.

This module implements Gaussian blur data augmentation for video action recognition.
"""

import random
from typing import Tuple, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GaussianBlur(BaseTransform):
    """Apply Gaussian Blur to images.

    This transform applies Gaussian blur to each frame in a video with a given
    probability. Gaussian blur is a common data augmentation technique that
    can help improve model robustness to out-of-focus or motion-blurred videos.

    Args:
        kernel_size (int): Size of the Gaussian kernel. Must be a positive odd number.
            Default: 5.
        sigma (float or tuple[float]): Standard deviation of the Gaussian kernel.
            If float, the fixed sigma will be used.
            If tuple (min, max), sigma will be randomly sampled from this range.
            Default: (0.1, 2.0).
        prob (float): Probability of applying Gaussian blur to the video.
            Default: 0.5.

    Examples:
        >>> # Apply Gaussian blur with random sigma
        >>> transform = dict(type='GaussianBlur', kernel_size=5, sigma=(0.1, 2.0), prob=0.3)
        >>> # Apply Gaussian blur with fixed sigma
        >>> transform = dict(type='GaussianBlur', kernel_size=7, sigma=1.5, prob=0.5)
    """

    def __init__(self,
                 kernel_size: int = 5,
                 sigma: Union[float, Tuple[float, float]] = (0.1, 2.0),
                 prob: float = 0.5):
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f'kernel_size must be a positive odd number, got {kernel_size}')

        if isinstance(sigma, (tuple, list)):
            if len(sigma) != 2:
                raise ValueError(f'sigma must be a number or a tuple of 2 numbers, got {sigma}')
            if sigma[0] > sigma[1]:
                raise ValueError(f'sigma[0] must be <= sigma[1], got {sigma}')
            if sigma[0] < 0 or sigma[1] < 0:
                raise ValueError(f'sigma values must be non-negative, got {sigma}')
        elif sigma < 0:
            raise ValueError(f'sigma must be non-negative, got {sigma}')

        if not 0 <= prob <= 1:
            raise ValueError(f'prob must be in [0, 1], got {prob}')

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    @cache_randomness
    def _random_params(self):
        """Get random parameters for Gaussian blur.

        Returns:
            tuple: (apply_blur, sigma) where apply_blur is a boolean indicating
            whether to apply blur, and sigma is the blur intensity.
        """
        apply_blur = random.random() < self.prob

        if isinstance(self.sigma, (tuple, list)):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma

        return apply_blur, sigma

    def transform(self, results: dict) -> dict:
        """Apply Gaussian blur to images.

        Required keys in results:
            - imgs: List of images (numpy arrays)

        Modified keys in results:
            - imgs: Blurred images if blur is applied
        """
        apply_blur, sigma = self._random_params()

        if not apply_blur:
            return results

        imgs = results['imgs']
        blurred_imgs = []

        for img in imgs:
            # Apply Gaussian blur using OpenCV
            # ksize: (width, height) of the kernel
            # sigmaX: standard deviation in X direction
            blurred = cv2.GaussianBlur(
                img,
                ksize=(self.kernel_size, self.kernel_size),
                sigmaX=sigma,
                sigmaY=sigma
            )
            blurred_imgs.append(blurred)

        results['imgs'] = blurred_imgs

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(kernel_size={self.kernel_size}, '
        repr_str += f'sigma={self.sigma}, '
        repr_str += f'prob={self.prob})'
        return repr_str
