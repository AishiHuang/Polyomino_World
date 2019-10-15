from src.networks import numpy_ffnet, dataset, analyses
import numpy as np


def main():
    np.set_printoptions(precision=4, suppress=True)

    hidden_size = 32
    learning_rate = 0.20
    num_epochs = 100
    weight_init = [0, 0.0000001]
    output_freq = 100
    verbose = False

    training_set = dataset.Dataset('data/w8-8_s9_c8_0_5_10.csv')
    test_set = dataset.Dataset('data/w8-8_s9_c8_0_1_10.csv')

    net = numpy_ffnet.NumpyFfnet(training_set.x_size, hidden_size, training_set.y_size, weight_init)
    costs = analyses.test(net, training_set)
    print("{:14s} Cost: {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(training_set.name, costs[0], costs[1], costs[2], costs[3]))

    analyses.train(net, training_set, test_set, num_epochs, learning_rate, output_freq)

    analyses.evaluate(net, test_set, verbose)

    analyses.evaluate(net, training_set, verbose)
    analyses.evaluate(net, test_set, verbose)

    net.save_weights("models/nff_h32_lr200_e1000_wn7.csv")


main()
