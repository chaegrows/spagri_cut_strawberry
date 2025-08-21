import rosbag2_py

bag_file = "/workspace/icra/cam_dir_9hour/icra_height_low/"

storage_options = rosbag2_py.StorageOptions(
    uri=bag_file, storage_id='sqlite3'  # storage_id를 강제 지정
)
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)