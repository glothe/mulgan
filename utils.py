import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from shutil import rmtree
import ffmpeg
from itertools import count, filterfalse

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
        save_image(denormalize_image(imbatch[i]), f"./videos/tmp/tmpim_{i}.png")
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
