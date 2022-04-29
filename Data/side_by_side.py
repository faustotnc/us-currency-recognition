from __future__ import annotations
from PIL import Image
import numpy as np
import streamlit as st
from os import path
import glob
import cv2
from copy import deepcopy

import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageOps

from smart_crop import smart_crop

# Create layout columns
st.set_page_config(layout="wide")

# Get the files
annotations_path = "./images/Annotations/"
images_path = "./images/JPEGImages/"

ann_files = sorted([f.split(path.sep)[-1] for f in glob.glob(
    f"{annotations_path}/*.xml") if path.isfile(f)])

# Saved Files
saved_imgs_path = "../Cleaned Data/images/JPEGImages"
saved_anns_path = "../Cleaned Data/images/Annotations"


# Select next image on button click
if 'current_file' not in st.session_state:
    st.session_state.current_file = 0


def show_prev_img():
    st.session_state.is_flipped_vert = False
    st.session_state.is_flipped_horz = False
    st.session_state.bbox_is_flipped_vert = False
    st.session_state.bbox_is_flipped_horz = False
    st.session_state.bbox_rotation = 0
    st.session_state.rotation = 0
    st.session_state.current_file -= 1

    if st.session_state.current_file < 0:
        st.session_state.current_file = 0


def show_next_img():
    st.session_state.is_flipped_vert = False
    st.session_state.is_flipped_horz = False
    st.session_state.bbox_is_flipped_vert = False
    st.session_state.bbox_is_flipped_horz = False
    st.session_state.bbox_rotation = 0
    st.session_state.rotation = 0
    st.session_state.current_file += 1

    if st.session_state.current_file >= len(ann_files):
        st.session_state.current_file = len(ann_files) - 1


def show_annotated_img(col, rcol, img, ann, title):
    draw = ImageDraw.Draw(img)
    draw.line((img.width/2, 0, img.width/2, img.height), fill=128, width=10)
    draw.line((0, img.height/2, img.width, img.height/2), fill=128, width=10)

    for obj in ann.findall("object"):
        box = obj.find("bndbox")
        x_min = int(float(box.find("xmin").text))
        x_max = int(float(box.find("xmax").text))
        y_min = int(float(box.find("ymin").text))
        y_max = int(float(box.find("ymax").text))

        shape = [(x_min, y_min), (x_max, y_max)]

        img1 = ImageDraw.Draw(img)
        img1.rectangle(shape, outline="#FAED27", width=10)

    col.image(img, caption=title+str(img.size), use_column_width=True)
    img = img.resize((640, 640))
    # img = ImageOps.pad(img, (640, 640))
    rcol.image(img, caption=title+str(img.size))


if 'is_flipped_vert' not in st.session_state:
    st.session_state.is_flipped_vert = False

if 'is_flipped_horz' not in st.session_state:
    st.session_state.is_flipped_horz = False

if 'rotation' not in st.session_state:
    st.session_state.rotation = 0

if 'bbox_rotation' not in st.session_state:
    st.session_state.bbox_rotation = 0


def flip_image_vert():
    st.session_state.is_flipped_vert = not st.session_state.is_flipped_vert


def flip_image_horz():
    st.session_state.is_flipped_horz = not st.session_state.is_flipped_horz


def rotate_image_right():
    st.session_state.rotation -= 90
    if st.session_state.rotation < -360:
        st.session_state.rotation = 0


def rotate_b_boxes_left():
    st.session_state.bbox_rotation += 90
    if st.session_state.bbox_rotation > 360:
        st.session_state.bbox_rotation = 0


if 'bbox_is_flipped_vert' not in st.session_state:
    st.session_state.bbox_is_flipped_vert = False

if 'bbox_is_flipped_horz' not in st.session_state:
    st.session_state.bbox_is_flipped_horz = False


def flip_b_boxes_vert():
    st.session_state.bbox_is_flipped_vert = not st.session_state.bbox_is_flipped_vert


def flip_b_boxes_horz():
    st.session_state.bbox_is_flipped_horz = not st.session_state.bbox_is_flipped_horz


# Divide the interface into columns
col1, col2, col3 = st.columns([2, 2, 1])

# Get the image name from the current filename
img_name = ann_files[st.session_state.current_file].split(".")[0]

# Open the current image and its associated annotations
ann = ET.parse(f"{annotations_path}{img_name}.xml")
raw_img = Image.open(f"{images_path}{img_name}.jpg")

# Rotate the image if specified
raw_img = raw_img.rotate(st.session_state.rotation, expand=True)

# Flip the image vertically if specified
if (st.session_state.is_flipped_vert):
    raw_img = ImageOps.flip(raw_img)

# Flip the image horizontally if specified
if (st.session_state.is_flipped_horz):
    raw_img = ImageOps.mirror(raw_img)

# Rotate the bounding boxes if specified
# Credit: https://stackoverflow.com/a/70804929/7407086
if st.session_state.bbox_rotation != 0:
    for obj in ann.findall("object"):
        box = obj.find("bndbox")
        x_min = round(float(box.find("xmin").text))
        y_min = round(float(box.find("ymin").text))
        x_max = round(float(box.find("xmax").text))
        y_max = round(float(box.find("ymax").text))

        # bb = np.array([x_min, y_min, x_max, y_max])
        bb = np.array([y_max, x_min, y_min, x_max])

        # Get all 4 coordinates of the box
        bb = np.array(((bb[0], bb[1]), (bb[2], bb[1]),
                      (bb[2], bb[3]), (bb[0], bb[3])))

        angle = st.session_state.bbox_rotation
        # Get the center of the image
        center = (raw_img.width//2, raw_img.height//2)
        # Get the rotation matrix, it's of shape 2x3
        rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Convert the array to [x,y,1] format to dot it with the rotMat
        bb_rotated = np.vstack((bb.T, np.array((1, 1, 1, 1))))

        # Perform Dot product and get back the points in shape of (4,2)
        bb_rotated = np.dot(rotMat, bb_rotated).T.astype(int)

        # Update the bounding box
        box.find("xmin").text = str(
            min(bb_rotated[(0, 0)], bb_rotated[(2, 0)]))
        box.find("ymin").text = str(
            min(bb_rotated[(0, 1)], bb_rotated[(1, 1)]))
        box.find("xmax").text = str(
            max(bb_rotated[(1, 0)], bb_rotated[(3, 0)]))
        box.find("ymax").text = str(
            max(bb_rotated[(2, 1)], bb_rotated[(3, 1)]))

# Flip the bbox's vertically if specified
if (st.session_state.bbox_is_flipped_vert):
    x_min, x_max = (2e100, 0)
    for obj in ann.findall("object"):
        box = obj.find("bndbox")

        new_x_min = raw_img.width - round(float(box.find("xmin").text))
        new_x_max = raw_img.width - round(float(box.find("xmax").text))

        box.find("xmin").text = str(min(new_x_min, new_x_max))
        box.find("xmax").text = str(max(new_x_min, new_x_max))

# Flip the bbox's horizontally if specified
if (st.session_state.bbox_is_flipped_horz):
    y_min, y_max = (2e100, 0)
    for obj in ann.findall("object"):
        box = obj.find("bndbox")

        new_y_min = raw_img.height - round(float(box.find("ymin").text))
        new_y_max = raw_img.height - round(float(box.find("ymax").text))

        box.find("ymin").text = str(min(new_y_min, new_y_max))
        box.find("ymax").text = str(max(new_y_min, new_y_max))

# Show the raw image
show_annotated_img(col1, col3, raw_img.copy(), ann, f"Before: ")

# Show the cropped image
new_img, new_ann = smart_crop(raw_img, deepcopy(ann.getroot()))
show_annotated_img(col2, col3, new_img.copy(), new_ann, f"After: ")


def save_files():
    '''Function to save the images and annotations'''

    # Save the image without EXIF data
    new_plain_img = Image.new(raw_img.mode, raw_img.size)
    new_plain_img.putdata(list(raw_img.getdata()))
    new_plain_img.save(f"{saved_imgs_path}/{img_name}.jpg")

    # Save the annotation file
    ann.write(f"{saved_anns_path}/{img_name}.xml")
    show_next_img()


def standard():
    rotate_image_right()
    rotate_b_boxes_left()
    rotate_b_boxes_left()
    flip_b_boxes_horz()


with st.sidebar:
    # Add buttons to the left-most column
    st.header(f"Image Name: {img_name}")
    st.write(f"Progress: {st.session_state.current_file} of {len(ann_files)}")
    st.progress(st.session_state.current_file/len(ann_files))

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.button("Prev Img", on_click=show_prev_img)
    c2.button("Next Img", on_click=show_next_img)
    c3.button("Save", on_click=save_files)

    st.button("Apply Standard Procedure", on_click=standard)

    st.write("-------------------")
    st.write("Image Settings")
    st.button("Flip Vert", key=1, on_click=flip_image_vert)
    st.button("Flip Horz", key=2, on_click=flip_image_horz)
    st.button("Rotate Right", key=3, on_click=rotate_image_right)

    # The buttons to edit the bounding boxes
    st.write("-------------------")
    st.write("BBox Settings")
    st.button("Flip Vert", key=4, on_click=flip_b_boxes_vert)
    st.button("Flip Horz", key=5, on_click=flip_b_boxes_horz)
    st.button("Rotate Left", key=6, on_click=rotate_b_boxes_left)
