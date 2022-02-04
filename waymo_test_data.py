import tensorflow as tf
import numpy as np

def view_raw_data():
    waymo_filenames = ['./data/waymo/training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord']
    waymo_raw_dataset = tf.data.TFRecordDataset(waymo_filenames)
    for record in waymo_raw_dataset.take(1):
        train = tf.train.Example()
        train.ParseFromString(record.numpy())
        print(train)

if __name__=='__main__':
    view_raw_data()