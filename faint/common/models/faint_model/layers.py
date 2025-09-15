from typing import List

import torch
import torch.nn as nn
import math
import einops
import timm

from faint.common.models.faint_model.theia_noprocessor import TheiaNoProcessor

class TimmEncoder(nn.Module):
    """
    Wrapper around timm models to extract the encoder output. The output shapes etc. have
    only been verified for efficientnet models.
    """
    def __init__(
        self,
        obs_encoder: str,
        pretrained: bool = True,
        freeze: bool = True,
        input_size: List[int] = [224, 224],
        num_channels: int = 3,
    ) -> None:
        super(TimmEncoder, self).__init__()

        if 'efficientnet' not in obs_encoder:
            raise NotImplementedError(
                'The settings below might not be valid for models other than efficientnet!'
            )

        self.encoder = timm.create_model(
            obs_encoder,
            pretrained=pretrained,
            in_chans=num_channels,
            num_classes=0,
            global_pool='',
        )

        if freeze:
            timm.utils.model.freeze(self.encoder)

        # Get the output token shape from the encoder
        with torch.no_grad():
            dummy_input = torch.zeros(
                (1, num_channels, input_size[0], input_size[1]),
                dtype=torch.float32,
            )
            out_shape = self.encoder.forward(dummy_input).shape

        self.encoder_dim = out_shape[1]
        self._num_patches = out_shape[-2] * out_shape[-1]
        self.out_size = out_shape[-2:]

    @property
    def encoding_dim(self):
        return self.encoder_dim
    
    @property
    def encoding_size(self):
        return self.out_size
    
    @property
    def num_patches(self):
        return self._num_patches

    def forward(
        self,
        img: torch.tensor,
    ) -> torch.Tensor:
        # get the observation encoding
        encoding = self.encoder.forward(img)

        # Reshape the encoding to [B, P, E] for consistency
        if len(encoding.shape) == 4:
            encoding = einops.rearrange(
                encoding,
                "b c h w -> b (h w) c",
            )
        else:
            raise NotImplementedError(
                f"""
                Encoder output shape was {encoding.shape}. Output shape validity hasn't been verified
                for encoders that output flattened sequences
                """
            )
        return encoding

class TheiaEncoder(nn.Module):

    """
    Wrapper around the Theia model to use it as an encoder.
    https://github.com/bdaiinstitute/theia/tree/main
    """

    theia_backbones = {
        "theaiinstitute/theia-tiny-patch16-224-cddsv": "facebook/deit-tiny-patch16-224",
        "theaiinstitute/theia-tiny-patch16-224-cdiv": "facebook/deit-tiny-patch16-224",
    }

    def __init__(
        self,
        obs_encoder: str,
        pretrained: bool = True,
        freeze: bool = True,
        input_size: List[int] = [224, 224],
    ) -> None:
        super(TheiaEncoder, self).__init__()

        # Load the Theia model with AutoProcessor stripped out
        self.encoder = TheiaNoProcessor(backbone=self.theia_backbones[obs_encoder])

        assert input_size[0] == input_size[1] == self.encoder.image_size, f"""
            Theia encoder expects {self.encoder.image_size}x{self.encoder.image_size} images
            """

        if pretrained:
            self.encoder.load_pretrained_weights(obs_encoder)
        if freeze:
            self.encoder.freeze_everything()

        # Get the output token shape from the encoder
        with torch.no_grad():
            dummy_input = torch.zeros(
                (1, 3, self.encoder.image_size, self.encoder.image_size),
                dtype=torch.float32,
            )
            out_shape = self.encoder.forward_feature(dummy_input).shape

        self.encoder_dim = out_shape[-1]
        self._num_patches = out_shape[-2]
        self.out_size = int(math.sqrt(self.num_patches))

    @property
    def encoding_dim(self):
        return self.encoder_dim
    
    @property
    def encoding_size(self):
        return [self.out_size, self.out_size]
    
    @property
    def num_patches(self):
        return self._num_patches

    def forward(
        self, img: torch.tensor,
    ) -> torch.Tensor:
        # get the observation encoding
        encoding = self.encoder.forward_feature(img)
        return encoding

class CatBlock(nn.Module):
    """
    Apply a transformer encoder to the
    concatenated observation and goal encodings
    with self-attention between all tokens.
    Only returns the tokens corresponding to the goal encoding
    """

    def __init__(
        self,
        input_dim,
        input_size,
        num_layers=4,
        nhead=4,
        ff_dim_factor=4,
        **kwargs,
        ):
        super(CatBlock, self).__init__()

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, 2*input_size[0]*input_size[1], input_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim*ff_dim_factor,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, obs_encoding, goal_encoding):
        """
        Args:
            obs_encoding: [B, P, E]
            goal_encoding: [B, P, E]
        Returns:
            x: [B, P, E]
        """
        # [B, P, E] -> [B, P*2, E]
        x = torch.cat((obs_encoding, goal_encoding), dim=1)
        x = x + self.position_embeddings

        # [B, P*2, E] -> [B, P*2, E]
        x = self.encoder(x)

        # Pick the goal tokens
        # [B, P*2, E] -> [B, P, E]
        x = x[:, obs_encoding.shape[1]:]
        return x

class CrossBlock(nn.Module):
    """
    Apply a transformer decoder to the goal encoding
    with cross-attention to the observation encoding.
    """

    def __init__(
        self,
        input_dim,
        input_size,
        num_layers=4,
        nhead=4,
        ff_dim_factor=4,
        **kwargs,
        ):
        super(CrossBlock, self).__init__()

        # Use learned position embeddings for now.
        # CroCo utilizes sinusoidal position embeddings
        # or Rope but skip that for now 
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, input_size[0]*input_size[1], input_dim)
        )

        # Add a register token for the goal encoding
        self.register_token = nn.Parameter(
            torch.zeros(1, 1, input_dim)
        )

        # Layer norm applied to the observation encoding
        # prior to cross-attention
        self.memory_norm = nn.LayerNorm(input_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim*ff_dim_factor,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

    def forward(self, obs_encoding, goal_encoding):
        """
        Args:
            obs_encoding: [B, P, E]
            goal_encoding: [B, P, E]
        Returns:
            x: [B, P, E]
        """

        # [1, 1, E] -> [B, 1, E]
        register_token = self.register_token.expand(goal_encoding.shape[0], -1, -1)

        x = goal_encoding + self.position_embeddings

        # Add the register token to the goal encoding
        x = torch.cat((x, register_token), axis=1)

        memory = self.memory_norm(obs_encoding + self.position_embeddings)

        x = self.decoder(x, memory)

        # Discard the register token
        x = x[:, :-1]
        return x

class ConvBlock(nn.Module):
    """
    Concatenates the observation and goal encodings channel-wise,
    reshapes the result into a 2D tensor, and applies a convolutional
    layer to aggregate information across the spatial dimensions.
    """
    def __init__(
        self,
        input_dim,
        input_size,
        kernel_size=3,
        stride=1,
        padding=1,
        **kwargs,
        ):
        super(ConvBlock, self).__init__()

        self.input_size = input_size

        # Build a series of 3 2D convolutional layers
        input_dims = [2*input_dim, input_dim, input_dim]
        modules = []
        for i_dim in input_dims:
            modules.extend([
                nn.Conv2d(
                    i_dim,
                    input_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(),
            ])
        self.conv = nn.Sequential(*modules)

    def forward(self, obs_encoding, goal_encoding):
        """
        Args:
            obs_encoding: [B, P, E]
            goal_encoding: [B, P, E]
        Returns:
            x: [B, P, E]
        """
        # [B, P, E] -> [B, P, E*2]
        x = torch.cat((obs_encoding, goal_encoding), dim=-1)

        # Reshape the input tensor to 2D for convolution
        x = einops.rearrange(
            x,
            "b (h w) c -> b c h w",
            h=self.input_size[0],
            w=self.input_size[1],
        )

        # [B, E*2, sqrt(P), sqrt(P)] -> [B, E, sqrt(P), sqrt(P)]
        x = self.conv(x)

        x = einops.rearrange(
            x,
            "b c h w -> b (h w) c",
        )
        return x

class NoOpBlock(nn.Module):
    """
    Simply returns the goal encoding without any modifications.
    """
    def __init__(self):
        super(NoOpBlock, self).__init__()

    def forward(self, obs_encoding, goal_encoding):
        """
        Args:
            obs_encoding: [B, P, E]
            goal_encoding: [B, P, E]
        Returns:
            x: [B, P, E]
        """
        return goal_encoding

class CompressionLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            input_size,
            compression_channels=2,
            output_dim=512,
            compression_type="flatten",
        ):
        """
        Compress a 2D feature map into a 1D tensor.

        Args:
            input_dim: Dimension of the input tensor
            input_size: Spatial size (H, W) of the input tensor
            compression_channels: Channels per spatial location in the output
            compression_type: Type of compression layer to use (flatten | mean)
        """

        super(CompressionLayer, self).__init__()

        self.input_size = input_size # Spatial size of the input / output

        if compression_type == "flatten":
            self.compression = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, compression_channels),
                nn.ReLU(True),
                nn.Flatten(),
            )
            self.output_dim = compression_channels * input_size[0] * input_size[1]

        elif compression_type == "mean":
            self.compression = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(input_dim, output_dim),
            )
            self.output_dim = output_dim
        else:
            raise ValueError(
                f"Invalid compression_type '{compression_type}'"
            )

    def forward(self, x):
        """
        Args:
            x: [B, P, E]
        Returns:
            x: 
        """
        # Reshape the input tensor to 2D for convolution
        x = einops.rearrange(
            x,
            "b (h w) c -> b c h w",
            h=self.input_size[0],
            w=self.input_size[1],
        )

        # [B, E, H, W] -> [B, self.output_dim]
        x = self.compression(x)
        return x

class ObsGoalFuser(nn.Module):
    def __init__(
            self,
            input_dim,
            input_size,
            fusion_block_type,
            compression_channels=2,
            compression_type="flatten",
            **kwargs,
        ):
        super(ObsGoalFuser, self).__init__()

        if fusion_block_type == "CatBlock":
            self.fusion_block = CatBlock(
                input_dim=input_dim,
                input_size=input_size,
                **kwargs,
            )
        elif fusion_block_type == "CrossBlock":
            self.fusion_block = CrossBlock(
                input_dim=input_dim,
                input_size=input_size,
                **kwargs,
            )
        elif fusion_block_type == "ConvBlock":
            self.fusion_block = ConvBlock(
                input_dim=input_dim,
                input_size=input_size,
                **kwargs,
            )
        elif fusion_block_type == "EarlyConv":
            self.fusion_block = NoOpBlock()
            
        elif fusion_block_type.lower() == "none":
            self.fusion_block = NoOpBlock()
        else:
            raise ValueError(
                f"Invalid obs goal fusion_block_type '{fusion_block_type}'"
            )
        
        self.compressor = CompressionLayer(
            input_dim=input_dim,
            input_size=input_size,
            compression_channels=compression_channels,
            compression_type=compression_type,
        )

    def forward(
            self,
            obs_encoding,
            goal_encoding,
        ):
        """
        Args:
            obs_encoding: [B, P, E]
            goal_encoding: [B, P, E]
        Returns:
            x: [B, E]
        """

        # [B, P, E], [B, P, E] -> [B, P, E]
        x = self.fusion_block(obs_encoding, goal_encoding)

        # [B, P, E] -> [B, E]
        x = self.compressor(x)
        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        """
        The basic positional encoding from ViNT.
        https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py

        Args:
            d_model: Dimension of the model
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        """
        Learned positional encoding.
        """
        super().__init__()

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, seq_len, d_model),
        )

    def forward(self, x):
        x = x + self.position_embeddings
        return x

class SeqEncoder(nn.Module):
    def __init__(
            self,
            seq_len,
            input_dim,
            num_layers=4,
            nhead=4,
            ff_dim_factor=4,
            output_type="cls",
            use_cls_token=True,
            pos_enc_type="learned",
        ):
        super(SeqEncoder, self).__init__()

        if output_type not in ["cls", "sequence"]:
            raise ValueError(
                f"Invalid output_type '{output_type}'"
            )
        if output_type == "cls" and not use_cls_token:
            raise ValueError(
                "output_type 'cls' requires use_cls_token=True"
            )
        if pos_enc_type not in ["learned", "sinusoidal"]:
            raise ValueError(
                f"Invalid pos_enc_type '{pos_enc_type}'"
            )

        self.output_type = output_type
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

        # Length of the input is observation sequence length + goal (+ cls_token)
        pos_enc_seq_len = seq_len + 1
        if use_cls_token:
            pos_enc_seq_len += 1
        
        if pos_enc_type == "learned":
            self.position_embeddings = LearnedPositionalEncoding(
                d_model=input_dim,
                seq_len=pos_enc_seq_len,
            )
        elif pos_enc_type == "sinusoidal":
            self.position_embeddings = SinusoidalPositionalEncoding(
                d_model=input_dim,
                seq_len=pos_enc_seq_len,
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim*ff_dim_factor,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        output_dim = input_dim
        if output_type == "sequence":
            output_dim *= seq_len + 1 # Seq_len + goal tokens
        self.output_dim = output_dim

    @property
    def encoding_dim(self):
        return self.output_dim

    def forward(
            self,
            obs_encoding,
            goal_encoding,
        ):
        """
        Args:
            obs_encoding: [B, S, E]
            goal_encoding: [B, E]
        Returns:
            x: [B, E]
        """
        
        x = [obs_encoding, goal_encoding.unsqueeze(1)]

        if self.use_cls_token:
            cls_token = einops.repeat(
                self.cls_token,
                "1 1 c -> b 1 c",
                b=obs_encoding.shape[0],
            )
            x.append(cls_token)

        # Concatenate the observation, goal, and cls tokens along the sequence dimension
        x = torch.cat(x, dim=1)

        x = self.position_embeddings(x)

        # [B, S+(1|2), E] -> [B, S+(1|2), E]
        x = self.encoder(x)

        if self.output_type == "cls":
            # Pick the cls token
            # [B, S+1, E] -> [B, E]
            x = x[:, -1]
        elif self.output_type == "sequence":
            # ViNT style flattens the entire sequence
            # Drop the cls token
            if self.use_cls_token:
                x = x[:, :-1]
            x = einops.rearrange(
                x,
                "b s e -> b (s e)",
            )
        return x