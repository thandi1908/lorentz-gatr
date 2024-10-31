from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_vector, extract_scalar
import torch


class Wrapper(torch.nn.Module):
    """Example wrapper around a L-GATr model.
    
    Expects input data that consists of a point cloud: one 4-momentum point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.
    
    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(self, blocks=10, hidden_mv_channels=16, hidden_s_channels=32):
        super().__init__()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        
    def forward(self, inputs):
        """Forward pass.
        
        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 4)
            4-momentum point cloud input data
        
        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        
        # Embed 4-momentum point cloud inputs in GA
        embedded_inputs = embed_vector(inputs).unsqueeze(-2)  # (..., num_points, 1, 16)
        
        # Pass data through GATr
        embedded_outputs, _ = self.gatr(embedded_inputs, scalars=None)  # (..., num_points, 1, 16)
        
        # Extract scalar outputs 
        outputs = extract_scalar(embedded_outputs)  # (..., 1)

        return outputs