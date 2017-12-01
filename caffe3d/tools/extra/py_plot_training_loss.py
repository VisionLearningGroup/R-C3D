#!/usr/bin/env python

import sys
import numpy as np
# Need to import the plotting package:
import matplotlib.pyplot as plt
import os

def main():

    if len(sys.argv) > 1:
        iter_loss_file = sys.argv[1]
    else:
        iter_loss_file = '/tmp/iter_loss.txt'

    if len(sys.argv) > 2:
        plot_file = sys.argv[2]
    else:
        plot_file = '/tmp/iter_loss.png'

    if len(sys.argv) > 3:
        iter_accuracy_file = sys.argv[3]
    else:
        iter_accuracy_file = '/tmp/iter_accuracy.txt'

    if len(sys.argv) > 4:
        iter_accuracy_top5_file = sys.argv[4]
    else:
        iter_accuracy_top5_file = '/tmp/iter_accuracy_top5.txt'

    print("[info] input iter_loss_file={}, iter_accuracy_file={}, "
          "output plot_file={}".format(
            iter_loss_file,
            iter_accuracy_file,
            plot_file
            )
          )

    # is accuracy available?
    if os.path.isfile(iter_accuracy_file) and \
            os.stat(iter_accuracy_file).st_size:
        accuracy_available = True
    else:
        accuracy_available = False

    #####################################################
    # LOSS

    # Read the file.
    f2 = open(iter_loss_file, 'r')
    # read the whole file into a single variable, which is a list of every row of the file.
    lines = f2.readlines()
    f2.close()

    # initialize some variable to be lists:
    x1 = []
    y1 = []

    # scan the rows of the file stored in lines, and put the values into some variables:
    for line in lines:
        p = line.split()
        x1.append(float(p[0]))
        y1.append(float(p[1]))
    xv = np.array(x1)
    yv = np.array(y1)

    if 'EPOCH' in os.environ:
        ep = os.environ['EPOCH']
        xv = xv / float(ep)
        xaxis_label = 'training epoch'
    else:
        xaxis_label = 'training iter'

    smooth = False
    if smooth:
        N = 4
        yv = np.convolve(yv, np.ones((N,))/N, mode='same')

    if accuracy_available:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
    else:
        fig, ax1 = plt.subplots()
    ax1.plot(xv, yv, 'bo-')
    #ax1.set_xlabel('training iteration (1 iter = 30 samples)')
    ax1.set_xlabel(xaxis_label)
    #ax1.grid(axis='x')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('training loss')
    #ax1.set_ylabel('training loss', color='b')
    #for tl in ax1.get_yticklabels():
    #    tl.set_color('b')
    ax1.set_ylim([0, 12])
    ax1.grid()

    plt.savefig(plot_file)

    #####################################################
    # ACCURACY

    if accuracy_available:
        ax2 = fig.add_subplot(212)

        # Read the file.
        f2 = open(iter_accuracy_file, 'r')
        # read the whole file into a single variable, which is a list of every row of the file.
        lines = f2.readlines()
        f2.close()

        # initialize some variable to be lists:
        x1 = []
        y1 = []

        # scan the rows of the file stored in lines, and put the values into some variables:
        for line in lines:
            p = line.split()
            x1.append(float(p[0]))
            y1.append(float(p[1]))
        xv = np.array(x1)
        yv = np.array(y1)
        accuracy = 100./np.exp(yv)

        if 'EPOCH' in os.environ:
            ep = os.environ['EPOCH']
            xv = xv / float(ep)
            xaxis_label = 'training epoch'
        else:
            xaxis_label = 'training iter'

        smooth = False
        if smooth:
            N = 4
            yv = np.convolve(yv, np.ones((N,))/N, mode='same')

        ax2.plot(xv, yv, 'bo-', label='top1')
        ax2.set_xlabel(xaxis_label)
        ax2.set_ylabel('test accuracy')
        ax2.grid()

        # top-5
        if os.path.isfile(iter_accuracy_top5_file):
            # Read the file.
            f2 = open(iter_accuracy_top5_file, 'r')
            # read the whole file into a single variable, which is a list of every row of the file.
            lines = f2.readlines()
            f2.close()

            # initialize some variable to be lists:
            y1 = []

            # scan the rows of the file stored in lines, and put the values into some variables:
            for line in lines:
                p = line.split()
                y1.append(float(p[1]))

            if len(y1) > 0:
                yv = np.array(y1)
                accuracy = 100./np.exp(yv)

                if smooth:
                    N = 4
                    yv = np.convolve(yv, np.ones((N,))/N, mode='same')

                ax2.plot(xv, yv, 'rx-', label='top5')
                ax2.legend(loc='best')

    plt.savefig(plot_file)

if __name__ == '__main__':
    main()
