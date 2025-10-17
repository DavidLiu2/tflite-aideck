# convert/rep_dataset.py
import glob, tensorflow as tf

REP_DIR = "../data/rep_images"
IMG_SIZE = (320, 320)

def rep_data_uint8_rgb():
    # Yields [1, 320, 320, 3] uint8 tensors
    for p in glob.glob(f"{REP_DIR}/*"):
        img = tf.io.decode_image(tf.io.read_file(p), channels=1)  # grayscale source
        img = tf.image.resize(img, IMG_SIZE, method="nearest")    # [H,W,1]
        img = tf.cast(img, tf.uint8)
        img = tf.tile(img, [1, 1, 3])                             # [H,W,3]
        img = tf.expand_dims(img, 0)                              # [1,H,W,3]
        yield [img]
