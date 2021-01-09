from polyomino_world.networks import dataset, network, analysis
import numpy as np


def main():
    np.set_printoptions(precision=4, suppress=True)
    hidden_size = 32
    hidden_actf = 'tanh'
    learning_rate = 0.20
    num_epochs = 1000
    weight_init = 0.00001
    output_freq = 25
    verbose = False
    x_type = 'WorldState' 
    y_type = 'FeatureVector' #'WorldState'
    included_features = [1, 1, 1, 0]  # Include: Shape, Size, Color, Action
    shuffle_sequences = True
    shuffle_events = False
    processor = 'CPU'
    optimizer = 'SGD'

    training_file = 'w8-8_s9_c7_0_100_1_train_omit_red_color.csv'
    # w8-8_s9_c1_0_100_1_second_train_one_color
    # w8-8_s9_c7_0_100_1_train_omit_one_color.csv
    # w8-8_s1_c8_0_100_1_second_train_one_shape
    # w8-8_s8_c8_0_100_1_train_omit_mono_shape
    # w8-8_s1_c8_0_100_1_second_train_mono_shape
    # w8-8_s8_c8_0_100_1_train_omit_domino_shape

    # w8-8_s9_c8_0_100_1_first_half_variant_train_(one_action)
    # w8-8_s9_c8_0_100_1_first_half_variant_train_(full_action)
    # w8-8_s9_c8_0_100_1_second_half_variant_train_(one_action)
    test_file = 'w8-8_s9_c8_0_10_1_complete_test.csv' 
    # 'w8-8_s9_c8_0_10_1_complete_test' 
    # 'w8-8_s9_c1_0_10_1_test_black_color.csv'
    # w8-8_s1_c8_0_10_1_test_mono_shape
    # w8-8_s1_c8_0_10_1_test_domino_shape

    # w8-8_s9_c8_0_10_1_second_half_variant_test(one_action)
    # w8-8_s9_c8_0_10_1_second_half_variant_test(full_action)
    network_directory = 'WS_FV_2021_1_8_0_32_57_omit_red_color_second_train_complete'
    # WS_FV_2020_12_28_17_0_28_omit_one_shape_second_train_(first_omit_second_complete)
    # WS_FV_2021_1_3_16_12_19_omit_one_color_second_train_(first_omit_second_complete)
    # WS_FV_2021_1_6_23_29_45_omit_mono_shape_second_train_omit_only
    # WS_FV_2021_1_6_23_29_45_omit_mono_shape_second_train_complete

    # WS_FV_2021_1_6_18_6_8_half_variant_second_train(one_action)_omit_only
    # WS_FV_2021_1_6_18_6_8_half_variant_second_train(one_action)_complete

    training_set = dataset.DataSet(training_file, None, included_features, processor)
    test_set = dataset.DataSet(test_file, None, included_features, processor)

    net = network.MlNet()
    # line 30 if starting a new model, line 33 if adding to an existing one
    # net.init_model(x_type, y_type, training_set,
    #                hidden_size, hidden_actf, optimizer, learning_rate, weight_init, processor)
    net.load_model(network_directory, included_features, processor)

    analysis.train_a(net, training_set, test_set, num_epochs, optimizer, learning_rate,
                     shuffle_sequences, shuffle_events, output_freq, verbose)


main()
