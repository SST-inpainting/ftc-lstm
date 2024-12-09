import os
import yaml
import time
import numpy as np
from math import exp
from typing import Tuple, List
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import wandb  # Weights & Biases for logging

#########################
# Dataset Class
#########################

class ChloraData(Dataset):
    """
    A PyTorch Dataset for loading sequential frames of data (e.g., chlorophyll data),
    optionally applying masks and transformations.
    """
    def __init__(self, 
                 root: str, 
                 frames: int, 
                 shape_scale=(48, 48),
                 is_month=False, 
                 mask_dir=None, 
                 is_train=True, 
                 train_test_ratio=0.7, 
                 task_name=None,
                 chunk_list=None,
                 mask_chunk_list=None,
                 global_mean=None,
                 global_std=None,
                 apply_log=True,
                 apply_normalize=True):
        """
        Args:
            root (str): Directory containing .npy files for data.
            frames (int): Number of frames (temporal dimension) per sample.
            shape_scale (tuple): If resizing is needed, (height, width).
            is_month (bool): Indicates if monthly mask is used.
            mask_dir (str): Directory containing corresponding mask npy files.
            is_train (bool): If True, dataset is for training; else for validation/testing.
            train_test_ratio (float): Ratio for splitting train/validation sets.
            task_name (str): Not fully utilized, can be used to specify a certain task (like 'dineof').
            chunk_list (list): Pre-computed list of data file paths grouped by frames.
            mask_chunk_list (list): Pre-computed list of mask file paths grouped by frames.
            global_mean (float): Global mean for normalization.
            global_std (float): Global std for normalization.
            apply_log (bool): Whether to apply log10 transform.
            apply_normalize (bool): Whether to apply normalization with provided mean/std.
        """
        super().__init__()
        self.root = root
        self.frames = frames
        self.shape_scale = shape_scale
        self.is_month = is_month
        self.is_train = is_train
        self.train_test_ratio = train_test_ratio
        self.task_name = task_name

        if self.is_month:
            assert mask_dir is not None, "mask_dir must be specified if is_month=True"
            self.mask_dir = mask_dir

        # If no external chunk list is provided, generate it
        if chunk_list is None:
            full_list = self._dir_list_select_combine(self.root, suffix='.npy')
            chunk_list = self._create_chunks(full_list, self.frames)
            chunk_list = self._train_test_split(chunk_list)
        self.chunk_list = chunk_list

        if self.is_month:
            if mask_chunk_list is None:
                full_mask_list = self._dir_list_select_combine(self.mask_dir, suffix='.npy')
                mask_chunk_list = self._create_chunks(full_mask_list, self.frames)
                mask_chunk_list = self._train_test_split(mask_chunk_list)
            self.mask_chunk = mask_chunk_list
        else:
            self.mask_chunk = []

        self.length = len(self.chunk_list)
        
        self.global_mean = global_mean
        self.global_std = global_std
        self.apply_log = apply_log
        self.apply_normalize = apply_normalize

        # Build transformation pipeline
        transform_list = [ToTensor()]
        if self.apply_log:
            # log10 requires positive values, handled by replacing zeros/nans before this step
            transform_list.append(Lambda(lambda x: torch.log10(torch.clamp(x, min=1e-8))))
        
        # If resizing is required:
        # transform_list.append(transforms.Resize(self.shape_scale))
        
        if self.apply_normalize and (self.global_mean is not None and self.global_std is not None):
            transform_list.append(transforms.Normalize(mean=[self.global_mean], std=[self.global_std]))
        
        self.transform = Compose(transform_list)

    def _train_test_split(self, chunk_list: List[List[str]]) -> List[List[str]]:
        np.random.shuffle(chunk_list)
        split_idx = int(self.train_test_ratio * len(chunk_list))
        return chunk_list[:split_idx] if self.is_train else chunk_list[split_idx:]

    @staticmethod
    def _dir_list_select_combine(base_dir: str, suffix: str='.npy') -> List[str]:
        files_list = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(suffix)]
        files_list.sort()
        return files_list

    @staticmethod
    def _create_chunks(file_list: List[str], frames: int) -> List[List[str]]:
        return [file_list[i:i+frames] for i in range(len(file_list)-frames+1)]

    def __len__(self):
        return self.length

    @staticmethod
    def _load_npy(file_path: str) -> np.ndarray:
        arr = np.load(file_path)
        return arr

    @staticmethod
    def _apply_mask_logic(data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # If mask is given (cloud mask), mask should align with data shape
        assert mask.shape == data.shape, f"Mask shape {mask.shape} doesn't match data {data.shape}"
        return data * mask, mask

    @staticmethod
    def _apply_random_mask(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = data.shape
        rand_mask = torch.from_numpy(np.where(np.random.normal(loc=100, scale=10, size=shape) < 90, 0, 1)).float()
        return data * rand_mask, rand_mask

    def __getitem__(self, index: int):
        chunk = self.chunk_list[index]
        if self.is_month:
            mask_index = min(index, len(self.mask_chunk)-1)
            mask_paths = self.mask_chunk[mask_index]
        else:
            mask_paths = [None]*len(chunk)

        inputs_list = []
        target_list = []
        masks_list = []

        for data_path, mask_path in zip(chunk, mask_paths):
            array = self._load_npy(data_path)
            array = np.nan_to_num(array, nan=1.0)
            array[array == 0.0] = 1.0
            
            data_tensor = self.transform(array)
            
            if self.is_month and mask_path is not None:
                mask_array = np.load(mask_path)
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
                in_data, mask_data = self._apply_mask_logic(data_tensor, mask_tensor)
            else:
                in_data, mask_data = self._apply_random_mask(data_tensor)

            inputs_list.append(in_data)
            target_list.append(data_tensor)
            masks_list.append(mask_data)

        inputs = torch.stack(inputs_list, dim=0)
        targets = torch.stack(target_list, dim=0)
        masks = torch.stack(masks_list, dim=0)
        
        return index, inputs.float(), targets.float(), masks.float()


#########################
# Model Building
#########################

def mkLayer(block_params: dict) -> nn.Sequential:
    """
    Given an OrderedDict-like structure of layer parameters, build a nn.Sequential block.
    The keys should contain layer type hints like 'conv', 'pool', 'deconv', 'relu', 'leaky'.
    """
    layers = []
    for layer_name, params in block_params.items():
        if 'pool' in layer_name:
            # expected params: [kernel_size, stride, padding]
            k, s, p = params
            layer = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            # expected params: [inC, outC, kernel_size, stride, padding]
            inC, outC, k, st, pad = params
            layer = nn.ConvTranspose2d(inC, outC, k, stride=st, padding=pad)
            layers.append((layer_name, layer))
        elif 'conv' in layer_name:
            # expected params: [inC, outC, kernel_size, stride, padding]
            inC, outC, k, st, pad = params
            layer = nn.Conv2d(inC, outC, k, stride=st, padding=pad)
            layers.append((layer_name, layer))
        else:
            raise NotImplementedError(f"Layer type not recognized in: {layer_name}")

        # Add activation if hinted in the layer name
        if 'relu' in layer_name:
            layers.append((f'relu_{layer_name}', nn.ReLU(inplace=True)))
        elif 'leaky' in layer_name:
            layers.append((f'leaky_{layer_name}', nn.LeakyReLU(negative_slope=0.2, inplace=True)))

    # Extract layers without names for nn.Sequential
    return nn.Sequential(*[layer for _, layer in layers])


class ConvLSTM_cell(nn.Module):
    """
    A ConvLSTM cell with optional frequency-domain convolution (fconv).
    """
    def __init__(self, shape, channels, kernel_size, features_num, fconv=False, frames_len=10, is_cuda=False):
        super().__init__()
        self.shape = shape
        self.channels = channels
        self.features_num = features_num
        self.kernel_size = kernel_size
        self.fconv = fconv
        self.padding = (kernel_size - 1) // 2
        self.frames_len = frames_len
        self.is_cuda = is_cuda

        groups_num = max(1, (4 * self.features_num) // 4)
        channel_num = 4 * self.features_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels + self.features_num, channel_num, self.kernel_size, padding=self.padding),
            nn.GroupNorm(groups_num, channel_num)
        )

        if fconv:
            self.semi_conv = nn.Sequential(
                nn.Conv2d(2 * (self.channels + self.features_num), channel_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num),
                nn.LeakyReLU(inplace=True)
            )
            self.global_conv = nn.Sequential(
                nn.Conv2d(8 * self.features_num, 4 * self.features_num, self.kernel_size, padding=self.padding),
                nn.GroupNorm(groups_num, channel_num)
            )

    def forward(self, inputs=None, hidden_state=None):
        hx, cx = self._init_hidden(inputs, hidden_state)
        output_frames = []

        for t in range(self.frames_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.channels, self.shape[0], self.shape[1], device=hx.device)
            else:
                x = inputs[t].to(hx.device)

            hy, cy = self._step(x, hx, cx)
            output_frames.append(hy)
            hx, cy = hy, cy

        return torch.stack(output_frames), (hy, cy)

    def _init_hidden(self, inputs, hidden_state):
        if hidden_state is not None:
            return hidden_state

        bsz = inputs.size(1) if inputs is not None else 1
        hx = torch.zeros(bsz, self.features_num, self.shape[0], self.shape[1], device='cuda' if self.is_cuda else 'cpu')
        cx = torch.zeros_like(hx)
        return hx, cx

    def _step(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        concat = torch.cat((x, hx), dim=1)
        gates_out = self.conv(concat)

        # Optional frequency domain convolution
        if self.fconv:
            # Fourier transform
            fft_dim = (-2, -1)
            freq = torch.fft.rfftn(concat, dim=fft_dim, norm='ortho')
            freq = torch.stack((freq.real, freq.imag), dim=-1)
            # Rearrange for convolution
            freq = freq.permute(0, 1, 4, 2, 3).contiguous()
            N, C, _, H, W2 = freq.size()
            freq = freq.view(N, -1, H, W2)
            ffc_conv = self.semi_conv(freq)
            # iFFT
            ifft_shape = ffc_conv.shape[-2:]
            ffc_out = torch.fft.irfftn(torch.complex(ffc_conv, torch.zeros_like(ffc_conv)), s=ifft_shape, dim=fft_dim, norm='ortho')
            # Resize to match gates_out size
            ffc_out_resize = F.interpolate(ffc_out, size=gates_out.size()[-2:], mode='bilinear', align_corners=False)
            combined = torch.cat((ffc_out_resize, gates_out), 1)
            gates_out = self.global_conv(combined)

        in_gate, forget_gate, hat_cell_gate, out_gate = torch.split(gates_out, self.features_num, dim=1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        hat_cell_gate = torch.tanh(hat_cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * hat_cell_gate)
        hy = out_gate * torch.tanh(cy)

        return hy, cy


class Encoder(nn.Module):
    """
    Encoder composed of multiple Conv/Pool layers and ConvLSTM cells in sequence.
    """
    def __init__(self, child_nets_params, convlstm_cells):
        super().__init__()
        assert len(child_nets_params) == len(convlstm_cells)
        self.block_num = len(child_nets_params)
        self.child_cells = nn.ModuleList([mkLayer(params) for params in child_nets_params])
        self.convlstm_cells = nn.ModuleList(convlstm_cells)

    def forward(self, inputs: torch.Tensor):
        # inputs shape: (batch, frames, C, H, W), need (frames, batch, C, H, W)
        inputs = inputs.transpose(0, 1)
        hidden_states = []

        # Forward with ConvLSTMs, flipping input each stage as per original code logic (if required)
        for i in range(self.block_num):
            if i > 0:
                inputs = torch.flip(inputs, [0])
            # Apply child cell (conv)
            fnum, bsz, ch, h, w = inputs.size()
            reshaped = inputs.reshape(-1, ch, h, w)
            processed = self.child_cells[i](reshaped)
            _, nch, nh, nw = processed.size()
            processed = processed.reshape(fnum, bsz, nch, nh, nw)

            # Apply ConvLSTM
            outputs, state_stage = self.convlstm_cells[i](processed, None)
            hidden_states.append(state_stage)
            inputs = outputs

        return tuple(hidden_states)


class Decoder(nn.Module):
    """
    Decoder composed of multiple ConvLSTM cells and UpConv (DeConv) layers to reconstruct frames.
    """
    def __init__(self, child_nets_params, convlstm_cells):
        super().__init__()
        assert len(child_nets_params) == len(convlstm_cells)
        self.block_num = len(child_nets_params)
        self.child_cells = nn.ModuleList([mkLayer(params) for params in child_nets_params])
        self.convlstm_cells = nn.ModuleList(convlstm_cells)

    def forward(self, hidden_states):
        # hidden_states are reversed for decoding
        hidden_states = hidden_states[::-1]
        inputs = None
        for i in range(self.block_num):
            # ConvLSTM cell (decoder)
            outputs, _ = self.convlstm_cells[i](inputs, hidden_states[i])
            seq_num, bsz, ch, h, w = outputs.size()
            reshaped = outputs.reshape(-1, ch, h, w)
            processed = self.child_cells[i](reshaped)
            _, nch, nh, nw = processed.size()
            inputs = processed.reshape(seq_num, bsz, nch, nh, nw)

        # final output shape: (frames, batch, C, H, W)
        return inputs.transpose(0, 1)


class ED(nn.Module):
    """
    ED model: Encoder-Decoder architecture that first encodes input frames 
    into hidden states and then decodes them back into predicted frames.
    """
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        state_list = self.encoder(inputs)
        output = self.decoder(state_list)
        return output


#########################
# SSIM & MSSIM Functions
#########################

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Compute SSIM
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    (_, channel, height, width) = img1.size()

    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=0, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding=0, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2*mu1_mu2 + C1)*v1)/((mu1_sq + mu2_sq + C1)*v2)

    if size_average:
        ret = ssim_map.mean()
        cs = cs.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
        cs = cs.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1)/2
        mcs = (mcs + 1)/2

    pow1 = mcs ** weights
    pow2 = ssims ** weights
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        channel = img1.size()[1]
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

#########################
# Early Stopping
#########################

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model_state, epoch, save_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, epoch, save_path)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model_state, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model_state, os.path.join(save_path, f"checkpoint_{epoch}_{val_loss:.6f}.pth.tar"))
        self.val_loss_min = val_loss


#########################
# Directory Setup
#########################

def record_dir_setting_create(father_dir: str, mark: str) -> str:
    dir_temp = os.path.join(father_dir, mark)
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)
    return dir_temp


#########################
# Main Training Procedure
#########################

def main(yaml_dir):
    
    with open(yaml_dir, "r") as aml:
        yml_params = yaml.safe_load(aml)

    # Extract parameters
    EPOCH       = yml_params["EPOCH"]
    BATCH       = yml_params["BATCH"]
    LR          = yml_params["LR"]
    frames      = yml_params["frames"]
    shape_scale = yml_params["shape_scale"]

    random_seed = yml_params["random_seed"]
    root_dir    = yml_params["root_dir"]
    mask_dir    = yml_params.get("mask_dir", None)
    is_month    = yml_params.get("is_month", False)
    device      = yml_params["device"]
    ckpt_files  = yml_params["ckpt_files"]

    fconv       = yml_params.get("fconv", False)
    min_cache   = yml_params.get("min_cache", False)
    beta        = yml_params.get("beta", 0.5)

    save_dir    = record_dir_setting_create(yml_params["ckpts_dir"], yml_params["mark"])
    stat_dir    = record_dir_setting_create(yml_params["stats_dir"], yml_params["mark"])
    img_dir     = record_dir_setting_create(yml_params["img_dir"], yml_params["mark"])
    log_dir     = record_dir_setting_create(yml_params["log_dir"], yml_params["mark"])

    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Initialize wandb
    wandb.init(
        project=yml_params.get("wandb_project", "ChloraModel"),
        entity=yml_params.get("wandb_entity", "your_wandb_username"),
        config=yml_params,
        name=yml_params.get("wandb_run_name", f"run_{int(time.time())}"),
        save_code=True
    )
    config = wandb.config

    # Prepare datasets and loaders
    train_dataset = ChloraData(root_dir, frames, shape_scale, is_month=is_month, mask_dir=mask_dir, is_train=True)
    valid_dataset = ChloraData(root_dir, frames, shape_scale, is_month=is_month, mask_dir=mask_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    # Define model parameters
    # Example parameters; replace with actual definitions
    # You can define these parameter sets based on your original model configuration
    # For demonstration, I'll create placeholder parameters
    from collections import OrderedDict

    # Example Encoder and Decoder Parameters
    convlstm_scs_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [1, 16, 3, 1, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
        ],
        [   
            ConvLSTM_cell(shape=(48,48), channels=16, kernel_size=5, features_num=64),
            ConvLSTM_cell(shape=(24,24), channels=64, kernel_size=5, features_num=96),
            ConvLSTM_cell(shape=(12,12), channels=96, kernel_size=5, features_num=96)
        ]
    ]

    convlstm_scs_decoder_params = [
        [
            OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
            OrderedDict({                                       
                'conv3_leaky_1': [64, 16, 3, 1, 1],
                'conv4_leaky_1': [16, 1, 1, 1, 0]
            }),
        ],
        [
            ConvLSTM_cell(shape=(12,12), channels=96, kernel_size=5, features_num=96),
            ConvLSTM_cell(shape=(24,24), channels=96, kernel_size=5, features_num=96),
            ConvLSTM_cell(shape=(48,48), channels=96, kernel_size=5, features_num=64),
        ]
    ]

    # Additional parameter sets (e.g., fconvlstm, min_cache) can be defined similarly
    # For brevity, we'll focus on the standard encoder/decoder

    # Initialize Encoder and Decoder
    encoder = Encoder(convlstm_scs_encoder_params[0], convlstm_scs_encoder_params[1]).to(device)
    decoder = Decoder(convlstm_scs_decoder_params[0], convlstm_scs_decoder_params[1]).to(device)

    # Initialize the ED model
    model = ED(encoder, decoder).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Load model checkpoint if provided
    current_epoch = 0
    if ckpt_files not in [None, 'None', '']:
        print('==> Loading existing model from checkpoint')
        checkpoint = torch.load(ckpt_files, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {current_epoch}")

    # Define loss functions
    ssmi_loss_func = SSIM().to(device)
    mae_loss_func = nn.SmoothL1Loss().to(device)
    mse_loss_func = nn.MSELoss().to(device)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=14, verbose=True)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # Training Loop
    for epoch in range(current_epoch, EPOCH + 1):
        model.train()
        train_losses = []
        t = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")

        for idx, inputs, targets, masks in t:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            
            # Compute losses
            mae_loss = mae_loss_func(pred, targets)
            mse_loss = mse_loss_func(pred, targets)
            ssmi_loss = ssmi_loss_func(pred[:, :, 0:1, :, :], targets[:, :, 0:1, :, :])  # Assuming first channel is relevant

            loss = beta * mse_loss + (1 - beta) * ((1 - ssmi_loss)/2)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()

            train_losses.append(loss.item())
            t.set_postfix(loss=loss.item())

        # Calculate average training loss
        mean_train_loss = np.mean(train_losses)
        wandb.log({"Train Loss": mean_train_loss, "epoch": epoch})

        # Validation every few epochs (e.g., every 10 epochs)
        if epoch % 10 == 0 and epoch != 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for idx, inputs, targets, masks in tqdm(valid_loader, desc=f"Validation Epoch {epoch}", leave=False):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    pred = model(inputs)

                    mae_loss = mae_loss_func(pred, targets)
                    mse_loss = mse_loss_func(pred, targets)
                    ssmi_loss = ssmi_loss_func(pred[:, :, 0:1, :, :], targets[:, :, 0:1, :, :])

                    loss = beta * mse_loss + (1 - beta) * ((1 - ssmi_loss)/2)
                    valid_losses.append(loss.item())

            mean_valid_loss = np.mean(valid_losses)
            wandb.log({"Validation Loss": mean_valid_loss, "epoch": epoch})
            print(f"[{epoch}/{EPOCH}] train_loss: {mean_train_loss:.6f} valid_loss: {mean_valid_loss:.6f}")

            # Step the scheduler
            scheduler.step(mean_valid_loss)

            # Prepare model state for checkpointing
            model_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # Early Stopping Check
            early_stopping(mean_valid_loss, model_state, epoch, save_dir)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # Log to wandb the model checkpoint as an artifact
            wandb.save(os.path.join(save_dir, f"checkpoint_{epoch}_{mean_valid_loss:.6f}.pth.tar"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    wandb.save(os.path.join(save_dir, "final_model.pth"))

    wandb.finish()

if __name__ == "__main__":
    # Load YAML configuration
    yaml_dir = r"/root/workspace/src/record/param/20231108.yml"
    main(yaml_dir)
