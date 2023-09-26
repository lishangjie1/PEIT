import torch.nn as nn
import torch
from .modules import VGG_FeatureExtractor, BidirectionalLSTM
from typing import Dict, List

class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, multi_line=False):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.multi_line = multi_line
        if not multi_line:
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1)) 
            
        else:
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 10)) 

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size
        

    def forward(self, input):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2)) # average on height dim
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
    def extract_features(self, input):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        
        if self.multi_line:
            # grid feature and skip RNN
            bz, channel, _,_ = visual_feature.shape
            visual_feature = visual_feature.reshape(bz, channel, -1)
            visual_feature = visual_feature.permute(0, 2, 1)
            """ Sequence modeling stage """
            contextual_feature = self.SequenceModeling(visual_feature)
            return contextual_feature # (Batch, width, hidden_state)
        else:
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2)) # average on height dim
            visual_feature = visual_feature.squeeze(3)
            """ Sequence modeling stage """
            contextual_feature = self.SequenceModeling(visual_feature)
            return contextual_feature # (Batch, width, hidden_state)
