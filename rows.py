import csv

path = r"F:/UNI/MT/new_pipeline/output_r1_t10/stats.csv"

with open(path) as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    for row in reader:
        frame = int(row[0])
        if frame >= 894 and frame <= 906:
            print(row[0], "kp", row[2], "good", row[3], "inl", row[4],
                  "mean", round(float(row[5]), 1),
                  "std", round(float(row[6]), 1),
                  "lap", round(float(row[7]), 1))