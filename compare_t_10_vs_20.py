import csv
import matplotlib.pyplot as plt

path20 = r"F:/UNI/MT/new_pipeline/output_r1/stats.csv"
path10 = r"F:/UNI/MT/new_pipeline/output_r1_t10_full/stats.csv"
out_png = r"F:/UNI/MT/new_pipeline/compare_t20_t10.png"


def load(path):
    frames = []
    kps = []
    inliers = []
    good = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            frames.append(int(row[0]))
            kps.append(int(row[2]))
            good.append(int(row[3]))
            inliers.append(int(row[4]))
    return frames, kps, good, inliers


def smooth(values, window):
    # moving average with plain loops
    result = []
    for i in range(len(values)):
        start = i - window
        if start < 0:
            start = 0
        total = 0.0
        count = 0
        for j in range(start, i + 1):
            total = total + values[j]
            count = count + 1
        result.append(total / count)
    return result


def summary(name, kps, good, inliers):
    total_good = 0
    total_inl = 0
    below_400 = 0
    below_100 = 0
    for i in range(len(kps)):
        total_good = total_good + good[i]
        total_inl = total_inl + inliers[i]
        if kps[i] < 400:
            below_400 = below_400 + 1
        if i > 0 and inliers[i] < 100:
            below_100 = below_100 + 1
    print(name,
          "| frames:", len(kps),
          "| mean good:", round(total_good / len(kps), 1),
          "| mean inliers:", round(total_inl / len(kps), 1),
          "| share:", round(total_inl / total_good, 3),
          "| kp<400:", below_400,
          "| inl<100:", below_100)


f20, kp20, g20, i20 = load(path20)
f10, kp10, g10, i10 = load(path10)

summary("T=20", kp20, g20, i20)
summary("T=10", kp10, g10, i10)

plt.figure(figsize=(11, 8))

plt.subplot(2, 1, 1)
plt.plot(f20, smooth(kp20, 25), label="T=20")
plt.plot(f10, smooth(kp10, 25), label="T=10")
plt.ylabel("keypoints (25-frame average)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(f20, smooth(i20, 25), label="T=20")
plt.plot(f10, smooth(i10, 25), label="T=10")
plt.xlabel("frame")
plt.ylabel("RANSAC inliers (25-frame average)")
plt.legend()

plt.savefig(out_png)
plt.close()
print("saved", out_png)