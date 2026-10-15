import numpy as np
import cv2
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

bag_path = r"new_pipeline\dataset-calib-cam7_512_16.bag"
typestore = get_typestore(Stores.ROS1_NOETIC)

with Reader(bag_path) as reader:

    # Step A: find the connection (stream) for the left camera
    cam0_connection = None
    for conn in reader.connections:
        if conn.topic == "/cam0/image_raw":
            cam0_connection = conn

    if cam0_connection is None:
        print("cam0 topic not found")
    else:
        # Step B: read the first message only
        first_msg = None
        for conn, timestamp, raw in reader.messages(connections=[cam0_connection]):
            first_msg = typestore.deserialize_ros1(raw, conn.msgtype)
            break

        # Step C: bytes -> 2D 16-bit image
        pixel_bytes = first_msg.data.tobytes()
        flat = np.frombuffer(pixel_bytes, dtype="<u2")
        img = flat.reshape(first_msg.height, first_msg.width)
        print("16-bit min/max:", img.min(), img.max())

        # Step D: 16-bit -> 8-bit (this was the missing part)
        lo = img.min()
        hi = img.max()
        img_f = img.astype(np.float32)
        img_f = img_f - lo
        img_f = img_f / (hi - lo)
        img_f = img_f * 255.0
        img8 = img_f.astype(np.uint8)
        print("8-bit:", img8.dtype, img8.min(), img8.max())

        # Step E: ORB on one frame
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(img8, None)

        print("keypoints found:", len(keypoints))
        print("descriptor array shape:", descriptors.shape)
        print("descriptor dtype:", descriptors.dtype)

        vis = cv2.drawKeypoints(img8, keypoints, None, color=(0, 255, 0))
        cv2.imwrite(r"new_pipeline/frame0_orb.png", vis)