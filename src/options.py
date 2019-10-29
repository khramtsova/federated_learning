
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Data distribution
    parser.add_argument('--dstr_Train', type=str, default="iid", help="Distribution of the training data"
                                                                      "can be: 'iid', 'non_iid' or 'non_iid_excl'")
    parser.add_argument('--n_labels_per_agent_Train', type=int, default=2,
                        help="Each agent have only n labels in it's training set")
    parser.add_argument('--sub_labels_Train', nargs='+', type=int, default=None,
                        help="Take into account only part of the labels")
    parser.add_argument('--dstr_Test', type=str, default="iid", help="Distribution of the test data"
                                                                     "can be: 'iid', 'non_iid' or 'non_iid_excl'")
    parser.add_argument('--n_labels_per_agent_Test', type=int, default=None,
                        help="Each agent have only n labels in it's test set")
    parser.add_argument('--sub_labels_Test', nargs='+', type=int, default=None,
                        help="Take into account only part of the labels")

    # Federated training
    parser.add_argument('--rounds', type=int, default=3, help="rounds of training")
    parser.add_argument('--num_workers', type=int, default=5, help="number of users")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs")
    parser.add_argument('--train_bs', type=int, default=32, help="local batch size")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")

    #Other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--log_folder', type=str, default="/logs/", help='Log folder from the root of the project')

    args = parser.parse_args()
    return args

