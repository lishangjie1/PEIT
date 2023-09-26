import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
with open(input_file, "r") as f, open(output_file, "w") as f1:
    for line in f:
        f1.write(' '.join(list(line.strip())) + "\n")

