import os


def rename_images_in_directory(directory_path):
    """
    Rename all images in the specified directory to sequential numbers.

    Parameters:
    - directory_path: str, path to the directory containing images.
    """

    files = os.listdir(directory_path)

    # filter out non-image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

    # renaming of images
    for index, image in enumerate(images, start=1):
        old_path = os.path.join(directory_path, image)
        new_filename = f"{index}{os.path.splitext(image)[1]}"  # Keep original extension
        new_path = os.path.join(directory_path, new_filename)

        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")


if __name__ == "__main__":
    directory_path = r""
    rename_images_in_directory(directory_path)
