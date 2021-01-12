import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from shutil import rmtree
import ffmpeg
from itertools import count, filterfalse
from math import sin, cos, pi
import torch


def normalize_image(image):
    return (image * 2 - 1).clamp(-1, 1)

def denormalize_image(image):
    return (image * .5 + .5).clamp(0, 1) 

def generate_sizes(max_size=250, min_size=25, scale_factor=0.75):
    max_size = 250
    size_factor = 1
    size = max_size
    sizes = [max_size]

    while size > min_size:
        size_factor *= 0.75
        size = int(size_factor * max_size)

        sizes.append(size)

    print(sizes)
    return sizes[::-1]

def display_image(image, ax=plt):
    ax.axis("off")
    ax.imshow(denormalize_image(image.cpu()).permute(1, 2, 0))

def display_images(imbatch):
    n_im = imbatch.shape[0]
    n_rows = n_im // 4 + 1
    fig, axes = plt.subplots(n_rows, 4, squeeze=False, figsize=(4 * 6.4, n_rows * 4.8))
    for r in range(n_rows):
        for c in range(4):
            axes[r][c].axis("off")
    for i in range(imbatch.shape[0]):
        display_image(imbatch[i], axes[i // 4][i % 4])

def save_as_video(imbatch, fps=24, outfile="movie_{}.mp4"):
    os.makedirs("./videos", exist_ok=True)
    rmtree('./videos/tmp', ignore_errors=True)
    os.makedirs("./videos/tmp", exist_ok=True)
    full_outfile = f"videos/{outfile}"
    out_fn_formatted = next(filterfalse(os.path.exists, map(full_outfile.format, count())))
    print(f"Writing video to {out_fn_formatted}")
    for i in range(imbatch.shape[0]):
        save_image(denormalize_image(imbatch[i]), f"./videos/tmp/tmpim_{i:04d}.png")
    try:
        ffmpeg\
            .input(f'./videos/tmp/tmpim_*.png', pattern_type="glob", framerate=fps)\
            .output(out_fn_formatted)\
            .run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    rmtree('./videos/tmp')

def cubic_interpolator(tensor_batch, freq, device):
    out_length = (tensor_batch.shape[0] - 1) * freq
    xs = range(tensor_batch.shape[0])
    interpolator = CubicSpline(xs, tensor_batch.cpu().numpy())
    new_noises = interpolator(torch.linspace(0, tensor_batch.shape[0] - 1, out_length))
    return torch.tensor(new_noises, device=device, dtype=torch.float32)

def linear_interpolator(tensor_batch, freq, device):
    # Just shift the input noises a step towards the left so that
    # the interpolations we want are between this_frame_in[i] and next_frame_in[i]
    out_length = (tensor_batch.shape[0] - 1) * freq
    this_frame_in = tensor_batch[:-1]
    next_frame_in = tensor_batch[1:]
    interpolated_noises = torch.zeros((freq, *this_frame_in.shape), device=device)
    for i, t in enumerate(torch.linspace(0, 1, freq)):
        interpolated_noises[i] = (1 - t) * this_frame_in + t * next_frame_in
    # interpolated_noises is now a tensor of all the frames with the first dimension corresponding to the t
    # We need to interleave the first two dimensions to get a batch of the right input noises
    interpolated_noises.transpose_(0, 1)
    return interpolated_noises.reshape(-1, *interpolated_noises.shape[2:])

def sine_interpolator(tensor_batch, freq, device):
    # Just shift the input noises a step towards the left so that
    # the interpolations we want are between this_frame_in[i] and next_frame_in[i]
    out_length = (tensor_batch.shape[0] - 1) * freq
    this_frame_in = tensor_batch[:-1]
    next_frame_in = tensor_batch[1:]
    interpolated_noises = torch.zeros((freq, *this_frame_in.shape), device=device)
    for i, t in enumerate(torch.linspace(0, 1, freq)):
        interpolated_noises[i] = cos(0.5 * pi * t) * this_frame_in + sin(0.5 * pi * t) * next_frame_in
    # interpolated_noises is now a tensor of all the frames with the first dimension corresponding to the t
    # We need to interleave the first two dimensions to get a batch of the right input noises
    interpolated_noises.transpose_(0, 1)
    return interpolated_noises.reshape(-1, *interpolated_noises.shape[2:])

def linear_norm_interpolator(tensor_batch, freq, device):
    out_length = (tensor_batch.shape[0] - 1) * freq
    this_frame_in = tensor_batch[:-1]
    start_norms = torch.norm(this_frame_in, dim=(2, 3))
    next_frame_in = tensor_batch[1:]
    end_norms = torch.norm(next_frame_in, dim=(2, 3))
    interpolated_noises = torch.zeros((freq, *this_frame_in.shape), device=device)
    for i, t in enumerate(torch.linspace(0, 1, freq)):
        interpolated_noises[i] = (1 - t) * this_frame_in + t * next_frame_in
        coef = ((1 - t) * start_norms + t * end_norms) / torch.norm(interpolated_noises[i], dim=(2, 3))
        interpolated_noises[i] *= coef[:, :, None, None]
    interpolated_noises.transpose_(0, 1)
    return interpolated_noises.reshape(-1, *interpolated_noises.shape[2:])

def linear_mean_std_interpolator(tensor_batch, freq, device):
    # Calculate summary statistics
    out_length = (tensor_batch.shape[0] - 1) * freq
    this_frame_in = tensor_batch[:-1]
    start_means = torch.mean(this_frame_in, dim=(2, 3))
    start_stds = torch.norm(this_frame_in - start_means[:, :, None, None], dim=(2, 3))
    next_frame_in = tensor_batch[1:]
    end_means = torch.mean(this_frame_in, dim=(2, 3))
    end_stds = torch.norm(this_frame_in - start_means[:, :, None, None], dim=(2, 3))

    interpolated_noises = torch.zeros((freq, *this_frame_in.shape), device=device)
    for i, t in enumerate(torch.linspace(0, 1, freq)):
        interpolated_noises[i] = (1 - t) * this_frame_in + t * next_frame_in
        mean = torch.mean(interpolated_noises[i], dim=(2, 3))
        reduced = interpolated_noises[i] - mean[:, :, None, None]
        std = torch.norm(reduced, dim=(2, 3))
        reduced = reduced / std[:, :, None, None]
        reduced *= (1 - t) * start_stds + t * end_stds
        reduced += (1 - t) * start_means + t * end_means
        interpolated_noises[i] = reduced
    interpolated_noises.transpose_(0, 1)
    return interpolated_noises.reshape(-1, *interpolated_noises.shape[2:])

