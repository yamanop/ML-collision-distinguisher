import csv
with open("combined.csv", "r") as f:
    for _ in range(5):
        print(f.readline())
