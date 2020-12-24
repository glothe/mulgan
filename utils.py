import matplotlib.pyplot as plt

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

def display_image(image):
    plt.imshow(denormalize_image(image.cpu()).permute(1, 2, 0))

def display_images(imbatch):
    for i in range(imbatch.shape[0]):
        display_image(imbatch[i])
