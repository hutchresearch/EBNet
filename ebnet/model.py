import torch
from enum import Enum
from functools import partial

get_dim_mults = lambda num_layers, conv_block_step: (
        2 ** ((torch.arange(num_layers) / conv_block_step).int())
    ).tolist()

class Activation(str, Enum):
    RELU = "relu"
    LEAKY = "leaky"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    ELU = "elu"
    PRELU = "prelu"
    GELU = "gelu"

def get_activation(act: Activation, inplace=True):
    if act == Activation.RELU:
        return partial(torch.nn.ReLU, inplace=inplace)
    elif act == Activation.LEAKY:
        return partial(torch.nn.LeakyReLU, inplace=inplace)
    elif act == Activation.TANH:
        return torch.nn.Tanh
    elif act == Activation.SIGMOID:
        return torch.nn.Sigmoid
    elif act == Activation.ELU:
        return partial(torch.nn.ELU, inplace=inplace)
    elif act == Activation.PRELU:
        return torch.nn.PReLU
    elif act == Activation.GELU:
        return torch.nn.GELU
    else:
        raise ValueError("Did not select valid activation function.")
    

class CrossDimConvBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels=64, groups=8, Act=torch.nn.ReLU):
        super(CrossDimConvBlock2D, self).__init__()
        self.conv_layer = torch.nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(1, in_channels),
            padding="same",
        )
        self.gn = torch.nn.GroupNorm(groups, out_channels)
        self.act = Act()
        self.pool = torch.nn.MaxPool2d(kernel_size=(1, in_channels))

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.act(out)
        out = self.pool(out)
        out = self.gn(out)

        return out

class ConvBlock1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, pooling_kernel=2, Act=torch.nn.ReLU):
        super(ConvBlock1D, self).__init__()
        self.conv_layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
        self.act = Act()
        self.pool = torch.nn.MaxPool1d(pooling_kernel)
    
    def forward(self, x):
        out = self.conv_layer(x)
        out = self.act(out)
        out = self.pool(out)
        return out

class ConvBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 1), pooling_kernel=(2, 1),  Act=torch.nn.ReLU):
        super(ConvBlock2D, self).__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu = Act()
        self.pool = torch.nn.MaxPool2d(pooling_kernel)
    
    def forward(self, x):
        out = self.conv_layer(x)
        out = self.relu(out)
        out = self.pool(out)
        return out


class ConvBackbone1D(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        dim_mults,
        kernel_size,
        pooling_kernel,
        groups,
        drop_p,
        Act=torch.nn.ReLU
    ):
        super(ConvBackbone1D, self).__init__()
        conv_blocks = []

        # Construct list of in channel/out channel tuples
        dims = [*map(lambda m: model_dim * m, dim_mults)]
        in_out_channels = list(zip(dims[:-1], dims[1:]))

        for in_channels, out_channels in in_out_channels:
            conv_block = ConvBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                pooling_kernel=pooling_kernel,
                Act=Act,
            )
            conv_blocks.append(conv_block)
        
        self.conv_blocks = torch.nn.Sequential(*conv_blocks)
        self.gn = torch.nn.GroupNorm(groups, out_channels)
        self.dropout = torch.nn.Dropout(drop_p)

    def forward(self, x):
        out = self.conv_blocks(x)
        out = self.gn(out)
        out = self.dropout(out)
        return out

class ConvBackbone2D(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        dim_mults,
        kernel_size,
        pooling_kernel,
        groups,
        drop_p,
        Act=torch.nn.ReLU
    ):
        super(ConvBackbone2D, self).__init__()
        conv_blocks = []

        # Construct list of in channel/out channel tuples
        dims = [*map(lambda m: model_dim * m, dim_mults)]
        in_out_channels = list(zip(dims[:-1], dims[1:]))

        for in_channels, out_channels in in_out_channels:
            conv_block = ConvBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                pooling_kernel=(pooling_kernel, 1),
                Act=Act,
            )
            conv_blocks.append(conv_block)
        
        self.conv_blocks = torch.nn.Sequential(*conv_blocks)
        self.gn = torch.nn.GroupNorm(groups, out_channels)
        self.dropout = torch.nn.Dropout(drop_p)

    def forward(self, x):
        out = self.conv_blocks(x)
        out = self.gn(out)
        out = self.dropout(out)
        return out

class LinearBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, drop_p: float, Act=torch.nn.Sigmoid) -> None:
        super(LinearBlock, self).__init__()
        self.layer = torch.nn.Linear(in_features, out_features)
        self.act = Act()            
        self.drop = torch.nn.Dropout(drop_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.act(x)
        x = self.drop(x)
        return x
        
class FeedForward(torch.nn.Module):
    def __init__(self, linear_dim: int, output_dim: int, num_layers:int, drop_p: float, Act=torch.nn.Sigmoid) -> None:
        super(FeedForward, self).__init__()

        self.layers = torch.nn.Sequential(
            *[
                torch.nn.LazyLinear(linear_dim),
                Act(),
                torch.nn.Dropout(drop_p),
                *[
                    LinearBlock(linear_dim, linear_dim, drop_p, Act=Act)
                    for _ in range(num_layers - 1)
                ],
                torch.nn.Linear(linear_dim, output_dim),
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EBModelPlus(torch.nn.Module):
    def __init__(self, 
        flux_in_channels: int=50,
        rv_in_channels: int=2,
        output_dim: int=46,
        kernel_size: int=2,
        flux_backbone_dim: int=64,
        num_flux_backbone_conv_layers: int=6,
        flux_conv_block_step: int=2,
        rv_backbone_dim: int=128,
        num_rv_backbone_conv_layers: int=3,
        rv_conv_block_step: int=1,
        meta_backbone_dim: int=128,
        num_meta_backbone_conv_layers: int=4,
        meta_conv_block_step: int=2,
        num_final_backbone_conv_layers: int=4,
        final_conv_block_step: int=2,
        num_linear_layers: int=6,
        linear_layer_dim: int=1024,
        drop_p: int=0.1,
        flux_act: Activation=Activation.RELU,
        rv_act: Activation=Activation.RELU,
        meta_act: Activation=Activation.RELU,
        backbone_act: Activation=Activation.RELU,
        feed_forward_act: Activation=Activation.SIGMOID,
    ) -> None:
        super(EBModelPlus, self).__init__()

        self.flux_in_channels = flux_in_channels
        self.rv_in_channels = rv_in_channels

        flux_dim_mults = get_dim_mults(num_flux_backbone_conv_layers, flux_conv_block_step)
        FluxAct = get_activation(flux_act)
        RVAct = get_activation(rv_act)
        MetaAct = get_activation(meta_act)
        BackboneAct = get_activation(backbone_act)
        FeedForwardAct = get_activation(feed_forward_act)

        self.flux_cross_dim_conv = CrossDimConvBlock2D(
            in_channels=flux_in_channels,
            out_channels=flux_backbone_dim,
            groups=8,
            Act=FluxAct,
        )
        self.flux_conv_backbone = ConvBackbone2D(
            model_dim=flux_backbone_dim,
            dim_mults=flux_dim_mults,
            kernel_size=kernel_size,
            pooling_kernel=kernel_size,
            groups=flux_backbone_dim,
            drop_p=drop_p,
            Act=FluxAct,
        )

        rv_dim_mults = get_dim_mults(num_rv_backbone_conv_layers, rv_conv_block_step)

        self.rv_cross_dim_conv = CrossDimConvBlock2D(
            in_channels=rv_in_channels,
            out_channels=rv_backbone_dim,
            groups=8,
            Act=RVAct,
        )
        self.rv_conv_backbone = ConvBackbone2D(
            model_dim=rv_backbone_dim,
            dim_mults=rv_dim_mults,
            kernel_size=kernel_size,
            pooling_kernel=kernel_size,
            groups=rv_backbone_dim,
            drop_p=drop_p,
            Act=RVAct,
        )

        meta_dim_mults = get_dim_mults(num_meta_backbone_conv_layers, meta_conv_block_step)
        self.meta_init_conv = ConvBlock1D(
            in_channels=1,
            out_channels=meta_backbone_dim,
            kernel_size=kernel_size,
            pooling_kernel=kernel_size,
            Act=MetaAct,
        )
        self.gn_meta = torch.nn.GroupNorm(8, meta_backbone_dim)
        self.meta_conv_backbone = ConvBackbone1D(
            model_dim=meta_backbone_dim,
            dim_mults=meta_dim_mults,
            kernel_size=kernel_size,
            pooling_kernel=kernel_size,
            groups=meta_backbone_dim,
            drop_p=drop_p,
            Act=MetaAct,
        )

        self.final_conv_model_dim = min(flux_backbone_dim * flux_dim_mults[-1], rv_backbone_dim * rv_dim_mults[-1])
        final_conv_dim_mults = get_dim_mults(num_final_backbone_conv_layers, final_conv_block_step)

        self.flatten = torch.nn.Flatten()
        self.final_conv_backbone = ConvBackbone2D(
            model_dim=self.final_conv_model_dim,
            dim_mults=final_conv_dim_mults,
            kernel_size=kernel_size,
            pooling_kernel=kernel_size,
            groups=self.final_conv_model_dim,
            drop_p=drop_p,
            Act=BackboneAct,
        )

        self.linear_layers = FeedForward(
            linear_dim=linear_layer_dim, 
            output_dim=output_dim, 
            num_layers=num_linear_layers, 
            drop_p=drop_p,
            Act=FeedForwardAct,
        )

    def forward(
        self, 
        flux: torch.Tensor, 
        rv: torch.Tensor, 
        meta: torch.Tensor, 
        period: torch.Tensor
    ) -> torch.Tensor:
        
        input_flux = flux.permute(0, 2, 1)[:, None, :, :]
        input_rv = rv.permute(0, 2, 1)[:, None, :, :]
        input_meta = meta[:, None, :]
        input_period = period #[:, None]
        
        assert input_flux.shape[-1] == self.flux_in_channels
        assert input_rv.shape[-1] == self.rv_in_channels
        batch_size = input_flux.shape[0]

        # Process input_flux
        x = self.flux_cross_dim_conv(input_flux)
        x = self.flux_conv_backbone(x)

        # Process input_rv
        y = self.rv_cross_dim_conv(input_rv)
        y = self.rv_conv_backbone(y)

        # Process input_meta
        z = self.meta_init_conv(input_meta)
        z = self.gn_meta(z)
        z = self.meta_conv_backbone(z)

        # Concatenate all layers
        x = x.view(batch_size, self.final_conv_model_dim, -1, 1)
        y = y.view(batch_size, self.final_conv_model_dim, -1, 1)
        out = self.final_conv_backbone(torch.cat((x, y), dim=2))

        out = self.flatten(out)
        z = self.flatten(z)
        combined = torch.cat((out, z, input_period), dim=1)
        result = self.linear_layers(combined)

        return result
    

class LoadedModelWrapper(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = torch.load(model_path)
        self.model.eval()  # Optional, but useful to ensure dropout/batchnorm behave correctly

    def forward(
        self, 
        flux: torch.Tensor, 
        rv: torch.Tensor, 
        meta: torch.Tensor, 
        period: torch.Tensor
    ) -> torch.Tensor:

        flux = flux.permute(0, 2, 1)[:, :, :, None]
        rv = rv.permute(0, 2, 1)[:, :, :, None]
        meta = meta[:, :, None]
        period_ = period[:, 0, None]
        parallax = period[:, 1, None]
        parallax_error = period[:, 2, None]
        return self.model(flux, rv, meta, period_, parallax, parallax_error)