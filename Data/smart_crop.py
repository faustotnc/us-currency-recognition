from PIL import MpoImagePlugin
from typing import Tuple
import xml.etree.ElementTree as ET


def compute_safe_area(axis_size: int, min_axis_size: int, padded_axis: Tuple[int, int]):
    # If the image is smaller than the min safe area in this axis, then there is nothing
    # else we can crop, so we return the original dimensions for this axis.
    if axis_size <= min_axis_size + 2:
        return (0, axis_size)

    # Represents the extra padding that must be added to `padded_axis` so
    # that the cropping area is of at least `min_size` in this axis.
    pad_area = padded_axis[1] - padded_axis[0]
    big_axis_delta = 0
    if (pad_area < min_axis_size):
        big_axis_delta = round((min_axis_size - pad_area) / 2)

    # If the object is too close to one of the image's edges, this
    # prevents the cropped area to go beyond the bounds of that edge.
    # This works because at this point we know that `min_axis_size`
    # is smaller than `axis_size`, so we should have enough space to crop
    # an area of `axis_size` along this axis.
    beta_axis = (
        # If the bounding box goes beyond the end of the axis, add extra
        # padding to the beginning of the axis.
        max(axis_size, (padded_axis[1] + big_axis_delta)) - axis_size,
        # If the bounding box goes beyond the start of the axis, add extra
        # padding to the end of the axis.
        abs(min(0, padded_axis[0] - big_axis_delta)),
    )

    # Return the safe area for cropping along this axis.
    new_min_axis = max(0, padded_axis[0]-big_axis_delta-beta_axis[0])
    new_max_axis = min(axis_size, padded_axis[1]+big_axis_delta+beta_axis[1])
    return (new_min_axis, new_max_axis)


def smart_crop(img: MpoImagePlugin.MpoImageFile, annotations: ET, min_size=None, padding=300):
    # Find the minimum area covers all objects in the image
    x_min, y_min, x_max, y_max = (2e100, 2e100, 0, 0)
    for obj in annotations.findall("object"):
        box = obj.find("bndbox")
        x_min = min(x_min, round(float(box.find("xmin").text)))
        y_min = min(y_min, round(float(box.find("ymin").text)))
        x_max = max(x_max, round(float(box.find("xmax").text)))
        y_max = max(y_max, round(float(box.find("ymax").text)))

    # The padded area that covers all bunding boxes in the image. In other
    # words, `delta_x` is the padded area of interest in the x-axis and
    # `delta_y` is the padded area of interest in the y-axis.
    delta_x = (max(0, x_min-padding), min(x_max+padding, img.width))
    delta_y = (max(0, y_min-padding), min(y_max+padding, img.height))

    # Set the minimum size for cropping (smallest, padded, hugging square as default)
    sz = max(delta_x[1]-delta_x[0], delta_y[1]-delta_y[0])
    min_size = (sz, sz) if not min_size else min_size

    # Compute the cropping area
    rho_x = compute_safe_area(img.width, min_size[0], delta_x)
    rho_y = compute_safe_area(img.height, min_size[1], delta_y)

    # Crop the image
    cropped = img.crop((rho_x[0], rho_y[0], rho_x[1], rho_y[1]))

    # Update all bounding boxes in the annotation XML
    for obj in annotations.findall("object"):
        box = obj.find("bndbox")
        box.find("xmin").text = str(
            round(float(box.find("xmin").text)) - rho_x[0])
        box.find("xmax").text = str(
            round(float(box.find("xmax").text)) - rho_x[0])
        box.find("ymin").text = str(
            round(float(box.find("ymin").text)) - rho_y[0])
        box.find("ymax").text = str(
            round(float(box.find("ymax").text)) - rho_y[0])

    # Return cropped image and updated annotation XML
    return (cropped, annotations)
