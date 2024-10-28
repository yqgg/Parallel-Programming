import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("-f", "--filename", type=str, default=None)
args = parser.parse_args()
figname = args.filename if args.filename else "runs/thread-times-0.txt"

def plot_speedup(filename):
    
    times = {}
    speedup = {}

    for line in open(file=filename, mode='r'):
        tokens = line.split(" ")
        n_thread = tokens[0]
        time_microsec  = tokens[1]
        times[int(n_thread)] = float(time_microsec)
    
    time_serial = times[1]

    for thread,time in times.items():
        speedup[thread] = float(f"{time_serial / time : .2f}")
    
    
    fig, axes = plt.subplots(1,1)
    axes.plot(speedup.keys(), speedup.values(), "--bo")
    axes.set_title("Perormance Scaling (N=1234567890)")
    axes.set_xlabel("Threads")
    axes.set_xticks(list(speedup.keys()))
    axes.set_ylabel("Speedup")
    fig.tight_layout()
    fig.savefig("speed-up.png")


if __name__ == "__main__":
    plot_speedup(figname)
