#!/usr/bin/env python3

"""
Train my neural network

Usage:
    homework4 autoencoder
    homework4 werk <partitions> <sampling>
    homework4 test

Arguments:
    autoencoder
        Run autoencoder

    werk
        Perform rap1 learning task with cross-validation

    <partitions>
        Number of partitions to make for cross-valitation

    <sampling>
        Sampling method for NN training input
        (slide) Iterate over sequence in 17nt sliding frame
        (space) Chop up each sequence into 17nt bits for inputs


Options:
    -n --normalize
        Normalize the raw scores from the alignment by the length of the
        shorter of the two sequences

"""
def werk():
    """
    Train neural network on RAP binding sites
        * Input layer with 17*4 (68) nodes + bias
        * Hidden layer with 23-35 nodes + bias
        * One output layer node

    Train against negative and positive binding sites
        * Import all negative sequences from .fa file
        * For each sequence, iterate every 17 bases and train with
          expected out of 0
        * Every 137 negative training instances, train against all
          positive binding sites with expected out of 1
        * Go until negative binding sites have been iterated through...
    """

    # Import Positive sequences
    positive_sites = [pos_seq.strip() for pos_seq in open('project_files/rap1-lieb-positives.txt')]

    # Import Negative sequences
    negative_sites = list(SeqIO.parse('project_files/yeast-upstream-1k-negative.fa', 'fasta'))

    # Separate into k random sections
    # Taken from : http://stackoverflow.com/questions/3352737/python-randomly-partition-a-list-into-n-nearly-equal-parts
    partitions = int(args['<partitions>'])
    neg_division = len(negative_sites) / float(partitions)
    neg_randomly_partitioned_list = [negative_sites[int(round(neg_division * i)): int(round(neg_division * (i + 1)))]
                                     for i in range(partitions)]

    pos_division = len(positive_sites) / float(partitions)
    pos_randomly_partitioned_list = [positive_sites[int(round(pos_division * i)): int(round(pos_division * (i + 1)))]
                                     for i in range(partitions)]

    # Cycle through negative sites subsets for cross-validation
    for index in range(int(args['<partitions>'])):
        # Set up cross-validation sets
        neg_site_list_copy = copy.deepcopy(neg_randomly_partitioned_list)
        del neg_site_list_copy[index]
        neg_site_training = [seq for partition in neg_site_list_copy for seq in partition]
        neg_cross_validation_set = neg_randomly_partitioned_list[index]

        pos_site_list_copy = copy.deepcopy(pos_randomly_partitioned_list)
        del pos_site_list_copy[index]
        pos_site_training = [seq for partition in pos_site_list_copy for seq in partition]
        pos_cross_validation_set = pos_randomly_partitioned_list[index]

        print("Training on Training Set...")

        # Set number of nodes
        NN = neural_network(68, 23, 1)

        # Initialize values
        NN.initialize_values()

        pos_counter = 0
        counter = 0

        if args['<sampling>'] == 'slide':
            for site in neg_site_training:

                # Iterate over site in 17nt sliding frames in negative sites
                for slice in range(len(site) - 16):
                    if site[slice:(slice + 17)].seq not in positive_sites:
                        NN.set_input_and_expected_values(site[slice:(slice + 17)].seq, autoencoder=False, negative=True)
                        NN.forward_propogation()
                        NN.backward_propogation()
                        NN.update_weights_and_bias()

                        pos_counter += 1

                    if pos_counter == len(pos_site_training):
                        for pos_site in pos_site_training:
                            NN.set_input_and_expected_values(pos_site, autoencoder=False, negative=False)
                            NN.forward_propogation()
                            NN.backward_propogation()
                            NN.update_weights_and_bias()

                        pos_counter = 0

                counter += 1

                print("Training set: {}/{} completed...".format(counter, len(neg_cross_validation_set)))

                max_change_1 = NN.matrix_1_errors.max()
                min_change_1 = NN.matrix_1_errors.min()
                max_change_2 = NN.matrix_2_errors.max()
                min_change_2 = NN.matrix_2_errors.min()

                if any([max_change_1 < 0.000001 and max_change_1 > 0,
                        min_change_1 > -.000001 and min_change_1 < 0]) or any(
                    [max_change_2 < 0.000001 and max_change_2 > 0,
                     min_change_2 > -0.000001 and min_change_2 < 0]):
                    print("Stop criterion met after {} iterations".format(counter))
                    break

        if args['<sampling>'] == 'space':
            for site in neg_site_training:
                # Chop sequence into 17nt blocks in negative sites
                number_of_blocks = int(len(site) / 17)

                for block in range(number_of_blocks):
                    if site[(block * 17):((block + 1) * 17)].seq not in positive_sites:
                        NN.set_input_and_expected_values(site[(block * 17):((block + 1) * 17)].seq, autoencoder=False, negative=True)
                        NN.forward_propogation()
                        NN.backward_propogation()
                        NN.update_weights_and_bias()

                        pos_counter += 1

                    if pos_counter == len(pos_site_training):
                        for pos_site in pos_site_training:
                            NN.set_input_and_expected_values(pos_site, autoencoder=False, negative=False)
                            NN.forward_propogation()
                            NN.backward_propogation()
                            NN.update_weights_and_bias()

                        pos_counter = 0

                    counter += 1

                max_change_1 = NN.matrix_1_errors.max()
                min_change_1 = NN.matrix_1_errors.min()
                max_change_2 = NN.matrix_2_errors.max()
                min_change_2 = NN.matrix_2_errors.min()

                if any([max_change_1 < 0.00000000001 and max_change_1 > 0,
                        min_change_1 > -.00000000001 and min_change_1 < 0]) or any(
                    [max_change_2 < 0.00000000001 and max_change_2 > 0,
                     min_change_2 > -0.00000000001 and min_change_2 < 0]):
                    print("Stop criterion met after {} iterations".format(counter))
                    break

        print("Performing Cross-validation")

        pos_list = []
        neg_list = []

        print("Negative cross-validation set...")
        counter = 0
        for site in neg_cross_validation_set:
            for slice in range(len(site) - 16):
                NN.set_input_and_expected_values(site[slice:slice + 17].seq, autoencoder=False, negative=True)
                NN.forward_propogation()
                neg_list.append(NN.output_layer_output)
            counter += 1
            print("Negative cross-validation: {}/{} completed...".format(counter, len(neg_cross_validation_set)))
            break

        print("Positive cross-validation set...")
        for site in pos_cross_validation_set:
            NN.set_input_and_expected_values(site, autoencoder=False)
            NN.forward_propogation()
            pos_list.append(NN.output_layer_output)

        print('Positive avg: {}'.format(sum(pos_list) / len(pos_list)))
        print('Negative avg: {}'.format(sum(neg_list) / len(neg_list)))
        print(NN.connection_matrix_1)
        print(NN.connection_matrix_2)

def autoencoder():
    NN = neural_network()
    NN.set_input_and_expected_values('GA', autoencoder=True)
    NN.initialize_values()

    # Stop criterion
    finished_working = False

    while finished_working == False:
        NN.forward_propogation()
        NN.backward_propogation()
        NN.update_weights_and_bias()

        max_change_1 = NN.matrix_1_errors.max()
        min_change_1 = NN.matrix_1_errors.min()
        max_change_2 = NN.matrix_2_errors.max()
        min_change_2 = NN.matrix_2_errors.min()

        if any([max_change_1 < 0.00001 and max_change_1 > 0,
                min_change_1 > -.00001 and min_change_1 < 0]) or any(
            [max_change_2 < 0.00001 and max_change_2 > 0,
             min_change_2 > -0.00001 and min_change_2 < 0]):
            finished_working = True

    print(NN.output_layer_output)

def test():
    test_sequences = open('project_files/rap1-lieb-test.txt')

if __name__ == '__main__':
    import docopt
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import seaborn as sns
    from Bio import SeqIO
    import sys
    import copy
    import pprint
    from .neural_network import neural_network

    args = docopt.docopt(__doc__)

    if args['autoencoder']:
        autoencoder()

    if args['werk']:
        werk()

    if args['test']:
        test()







