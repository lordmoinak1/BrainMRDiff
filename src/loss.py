import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from gudhi import CubicalComplex
from gudhi.wasserstein import wasserstein_distance


def soft_ed_transform(image: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """
    Compute a soft approximation of the Euclidean Distance Transform (EDT)
    using Gaussian blurring.

    Args:
        image (torch.Tensor): Binary image tensor of shape (N, 1, H, W)
        sigma (float): Standard deviation for Gaussian blur, controls smoothness.

    Returns:
        torch.Tensor: Differentiable soft approximation of EDT.
    """
    # Ensure input is float
    image = image.float()

    # Create a Gaussian kernel
    kernel_size = int(6 * sigma) + 1  # Ensure odd kernel size
    kernel_1d = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    
    # Convert to 2D Gaussian kernel
    kernel_2d = torch.ger(kernel_1d, kernel_1d).to(image.device)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, k, k)

    # Apply Gaussian blur (2D convolution)
    padded_image = F.pad(image, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
    edt_soft = F.conv2d(padded_image, kernel_2d, padding=0)
    
    return edt_soft

class SATopologicalLoss(nn.Module):
    def __init__(self, deblurr_kernel=None):
        """
        Initialize the topological loss function.
        Args:
            mask (torch.Tensor): Binary mask tensor for cropping (shape: H x W).
            deblurr_kernel (torch.Tensor): Kernel for deblurring (e.g., Gaussian).
        """
        super(SATopologicalLoss, self).__init__()
        self.deblurr_kernel = deblurr_kernel
        # self.deblurr_kernel = cv2.medianBlur(np.rand(128, 128), 5)


    def deblur(self, image):
        """Apply deblurring to the image using a convolution."""
        if self.deblurr_kernel is not None:
            image = torch.nn.functional.conv2d(
                image.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                self.deblurr_kernel.unsqueeze(0).unsqueeze(0),
                padding='same'
            ).squeeze()  # Remove extra dimensions
        return image

    def compute_persistence_diagram(self, data):
        """Compute the persistence diagram of a 2D array."""
        cubical_complex = CubicalComplex(dimensions=data.shape, top_dimensional_cells=data.flatten())
        cubical_complex.compute_persistence()
        return cubical_complex.persistence()#cubical_complex.persistence_diagram()

    def forward(self, noise, noise_pred, tumor, brain, wmt, cgm, lv):
        """
        Compute the topological loss between noise and noise_pred.
        Args:
            noise (torch.Tensor): Ground truth noise (shape: H x W).
            noise_pred (torch.Tensor): Predicted noise (shape: H x W).
        Returns:
            torch.Tensor: Topological loss value.
        """
        # Step 1: Deblur noise and noise_pred
        noise_deblurred = self.deblur(noise)
        noise_pred_deblurred = self.deblur(noise_pred)

        # Step 2: Apply the mask
        noise_tumor_mask = noise_deblurred * tumor
        noise_pred_tumor_mask = noise_pred_deblurred * tumor

        #### Step: EDT ####
        noise_tumor_mask = soft_ed_transform(noise_tumor_mask)
        noise_pred_tumor_mask = soft_ed_transform(noise_pred_tumor_mask)

        # Step 3: Compute persistence diagrams
        pd_noise_tumor = self.compute_persistence_diagram(noise_tumor_mask.cpu().detach().numpy())
        pd_noise_pred_tumor = self.compute_persistence_diagram(noise_pred_tumor_mask.cpu().detach().numpy())

        # a_tumor = np.array([[birth, death] for dim, (birth, death) in pd_noise_tumor if dim == 0 and death != np.inf])
        # b_tumor = np.array([[birth, death] for dim, (birth, death) in pd_noise_pred_tumor if dim == 0 and death != np.inf])

        a_tumor = np.array([[birth, death] for dim, (birth, death) in pd_noise_tumor if dim == 1 and death != np.inf])
        b_tumor = np.array([[birth, death] for dim, (birth, death) in pd_noise_pred_tumor if dim == 1 and death != np.inf])

        # Step 4: Compute Wasserstein distance as the topological loss
        topo_loss = wasserstein_distance(a_tumor, b_tumor)
        
        return torch.tensor(topo_loss, dtype=torch.float32)

if __name__ == "__main__":
    flag = 0

    noise = torch.randn(2, 1, 128, 128)
    noise_pred = torch.randn(2, 1, 128, 128)
    mask = torch.randn(2, 1, 128, 128)

    loss = SATopologicalLoss()

    outputs = loss(noise, noise_pred, mask, mask, mask, mask, mask)

    print(outputs)




