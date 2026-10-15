import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

# ---------------- settings ----------------
BAG_PATH = r"F:\UNI\MT\new_pipeline\dataset-room1_512_16.bag"
TOPIC = "/cam0/image_raw"
OUT_DIR = r"F:/UNI/MT/new_pipeline/output_r1_t10_full"
N_FEATURES = 1000
RATIO = 0.75
FAST_T = 10    # OpenCV default is 20; try 10 and 5 on weak frames
SAVE_EVERY = 25        # save a match picture every N frames
START_FRAME = 0      # skip frames before this (for testing a window)
MAX_FRAMES = None    # stop at this frame index (None = all frames)
WEAK_KP = 400          # a frame with fewer keypoints than this is "weak"
SAVE_RAW_FRAMES = [900, 1049, 1050]   # raw frames to save for visual checks
# ------------------------------------------


def message_to_image(msg):
    flat = np.frombuffer(msg.data.tobytes(), dtype="<u2")
    img16 = flat.reshape(msg.height, msg.width)
    return img16


def to_uint8(img16):
    img_f = img16.astype(np.float32)
    img_f = img_f / 257.0
    img8 = img_f.astype(np.uint8)
    return img8


def ratio_test_matches(matcher, desc_prev, desc_curr):
    good = []
    pairs = matcher.knnMatch(desc_prev, desc_curr, k=2)
    for pair in pairs:
        if len(pair) < 2:
            continue
        best = pair[0]
        second = pair[1]
        if best.distance < RATIO * second.distance:
            good.append(best)
    return good


def ransac_filter(kp_prev, kp_curr, good):
    # RANSAC needs at least 8 points for the fundamental matrix
    if len(good) < 8:
        return []

    pts_prev = np.zeros((len(good), 2), dtype=np.float32)
    pts_curr = np.zeros((len(good), 2), dtype=np.float32)
    for i in range(len(good)):
        pts_prev[i] = kp_prev[good[i].queryIdx].pt
        pts_curr[i] = kp_curr[good[i].trainIdx].pt

    F, mask = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return []

    mask = mask.ravel()
    inliers = []
    for i in range(len(good)):
        if mask[i] == 1:
            inliers.append(good[i])
    return inliers


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)

    print("BAG_PATH:", BAG_PATH)
    print("file exists:", os.path.exists(BAG_PATH))
    print("FAST threshold:", FAST_T, "| ratio:", RATIO, "| features:", N_FEATURES)

    orb = cv2.ORB_create(nfeatures=N_FEATURES, fastThreshold=FAST_T)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)   # Hamming because ORB is binary

    # each row: frame, timestamp, keypoints, ratio_good, inliers, mean, std, lap_var
    rows = []
    prev_img8 = None
    prev_kp = None
    prev_desc = None

    with Reader(BAG_PATH) as reader:
        cam0_connection = None
        for conn in reader.connections:
            if conn.topic == TOPIC:
                cam0_connection = conn

        if cam0_connection is None:
            print("topic not found")
            return

        frame_index = 0
        for conn, timestamp, raw in reader.messages(connections=[cam0_connection]):
            if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
                break

            if frame_index < START_FRAME:
                frame_index = frame_index + 1
                continue

            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            img8 = to_uint8(message_to_image(msg))

            # ----- image quality measures (for the weak-frame diagnosis) -----
            mean_val = float(img8.mean())          # overall brightness
            std_val = float(img8.std())            # contrast
            lap_var = float(cv2.Laplacian(img8, cv2.CV_64F).var())   # sharpness

            if frame_index in SAVE_RAW_FRAMES:
                cv2.imwrite(OUT_DIR + "/raw_" + str(frame_index) + ".png", img8)

            kp, desc = orb.detectAndCompute(img8, None)
            n_kp = len(kp)

            n_good = 0
            n_inliers = 0

            if prev_desc is not None and desc is not None:
                good = ratio_test_matches(matcher, prev_desc, desc)
                inliers = ransac_filter(prev_kp, kp, good)
                n_good = len(good)
                n_inliers = len(inliers)

                if frame_index % SAVE_EVERY == 0 and n_inliers > 0:
                    to_draw = inliers[:60]
                    vis = cv2.drawMatches(prev_img8, prev_kp, img8, kp, to_draw, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imwrite(OUT_DIR + "/match_" + str(frame_index) + ".png", vis)

            rows.append([frame_index, timestamp, n_kp, n_good, n_inliers,
                         mean_val, std_val, lap_var])

            if frame_index % 25 == 0:
                print("frame", frame_index, "| kp", n_kp, "| ratio-good", n_good,
                      "| inliers", n_inliers)

            prev_img8 = img8
            prev_kp = kp
            prev_desc = desc
            frame_index = frame_index + 1

    # =====================================================================
    # everything below runs AFTER the bag is closed
    # =====================================================================
    if len(rows) == 0:
        print("no frames processed")
        return

    # ---------- overall statistics ----------
    total_good = 0
    total_inl = 0
    total_kp = 0
    min_kp = rows[0][2]
    max_kp = rows[0][2]
    low_inlier_count = 0
    weak_frames = []

    for row in rows:
        total_good = total_good + row[3]
        total_inl = total_inl + row[4]
        total_kp = total_kp + row[2]
        if row[2] < min_kp:
            min_kp = row[2]
        if row[2] > max_kp:
            max_kp = row[2]
        if row[0] > 0 and row[4] < 100:
            low_inlier_count = low_inlier_count + 1
        if row[2] < WEAK_KP:
            weak_frames.append(row[0])

    print("mean ratio-good per frame:", total_good / len(rows))
    print("mean inliers per frame:", total_inl / len(rows))
    if total_good > 0:
        print("overall inlier share:", total_inl / total_good)
    print("keypoints min/mean/max:", min_kp, total_kp / len(rows), max_kp)
    print("frames with fewer than", WEAK_KP, "keypoints:", len(weak_frames))
    print("frames with fewer than 100 inliers:", low_inlier_count)

    # ---------- weak vs other: what is different about the weak frames? ----------
    weak_n = 0
    weak_mean = 0.0
    weak_std = 0.0
    weak_lap = 0.0
    ok_n = 0
    ok_mean = 0.0
    ok_std = 0.0
    ok_lap = 0.0

    for row in rows:
        if row[2] < WEAK_KP:
            weak_n = weak_n + 1
            weak_mean = weak_mean + row[5]
            weak_std = weak_std + row[6]
            weak_lap = weak_lap + row[7]
        else:
            ok_n = ok_n + 1
            ok_mean = ok_mean + row[5]
            ok_std = ok_std + row[6]
            ok_lap = ok_lap + row[7]

    if weak_n > 0:
        print("WEAK  frames (n=" + str(weak_n) + ") mean/std/lap_var:",
              round(weak_mean / weak_n, 2), round(weak_std / weak_n, 2),
              round(weak_lap / weak_n, 2))
    else:
        print("no weak frames in this window")

    if ok_n > 0:
        print("OTHER frames (n=" + str(ok_n) + ") mean/std/lap_var:",
              round(ok_mean / ok_n, 2), round(ok_std / ok_n, 2),
              round(ok_lap / ok_n, 2))

    # ---------- save CSV ----------
    with open(OUT_DIR + "/stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_ns", "keypoints", "ratio_good",
                         "inliers", "mean", "std", "lap_var"])
        for row in rows:
            writer.writerow(row)

    # ---------- plot ----------
    frames = []
    kps = []
    inl = []
    for row in rows:
        frames.append(row[0])
        kps.append(row[2])
        inl.append(row[4])

    plt.figure(figsize=(10, 4))
    plt.plot(frames, kps, label="keypoints")
    plt.plot(frames, inl, label="RANSAC inliers")
    plt.xlabel("frame")
    plt.ylabel("count")
    plt.legend()
    plt.savefig(OUT_DIR + "/stats.png")
    plt.close()

    print("frames processed:", len(rows))
    print("done. outputs in", OUT_DIR)


if __name__ == "__main__":
    main()