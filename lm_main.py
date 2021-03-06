from comet_ml import Experiment
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
import argparse
from tqdm import tqdm  # optional progress bar
from ignite.metrics import Precision, Recall

from lstm import *
from AGNewsDataset import *

# TODO: Set hyperparameters
hyperparams = {
    "rnn_size": 500,  # assuming encoder and decoder use the same rnn_size
    "num_epochs": 3,
    "batch_size": 20,
    "learning_rate": 1e-5,
    "window_size": 50
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on: ", device)


def train(model, train_loader, experiment, hyperparams):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # TODO: Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    model = model.train()
    with experiment.train():
        # TODO: Write training loop
        for e in range(hyperparams['num_epochs']):
            for batch in tqdm(train_loader):
                inputs = batch['lm_input']
                labels = batch['lm_label']

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                y_pred = model(inputs)
                loss = loss_fn(torch.transpose(y_pred, 1, 2), labels)
                loss.backward()
                optimizer.step()


def test(model, test_loader, experiment, hyperparams):
    """
    Validates the model performance as LM on never-seen data.
    :param model: the trained model to use for prediction
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    :param hyperparams: hyperparameters dictionary
    :param bpe: is bpe dataset or not
    """
    # TODO: Define loss function, total loss, and total word count
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction = "sum")
    total_loss = 0
    word_count = 0
    total_correct = 0
    total_words = 0

    precision = Precision(average=False, is_multilabel=True)# , device = device)
    recall = Recall(average=False, is_multilabel=True)# , device = device)

    model = model.eval()
    with experiment.test():
        # TODO: Write testing loop
        with torch.no_grad():
            for batch in tqdm(test_loader):

                inputs = batch['lm_input']
                labels = batch['lm_label']
                word_count = sum([len(arr) for arr in inputs])

                inputs = inputs.to(device)
                labels = labels.to(device)

                y_pred = model(inputs)
                loss = loss_fn(torch.transpose(y_pred, 1, 2), labels)

                total_loss += loss

                prbs = torch.nn.functional.softmax(y_pred , dim = 2)
                batch_acc, batch_correct, batch_words = accuracy_func(prbs, labels, word_count)

                total_correct += batch_correct
                total_words += batch_words

                predicted_ids = torch.argmax(prbs, dim=2)
                y_pred_one_hot = F.one_hot(predicted_ids.to(torch.int64), num_classes=prbs.shape[-1])
                labels_one_hot = F.one_hot(labels.to(torch.int64), num_classes=prbs.shape[-1])
                precision.update((torch.transpose(y_pred_one_hot, 1, 2), torch.transpose(labels_one_hot, 1, 2)))
                recall.update((torch.transpose(y_pred_one_hot, 1, 2), torch.transpose(labels_one_hot, 1, 2)))

        perplexity = torch.exp(total_loss / total_words)
        accuracy = total_correct / total_words

        perplexity = perplexity.item()
        # accuracy = accuracy.cpu()

        
        F1 = (precision * recall * 2 / (precision + recall)).mean().compute().item()
        print("F1: ", F1)
        experiment.log_metric("F1", F1)

        print("total_correct: ", total_correct)
        print("total_words: ", total_words)

        print("perplexity:", perplexity)
        print("accuracy:", accuracy)
        experiment.log_metric("perplexity", perplexity)
        experiment.log_metric("accuracy", accuracy)

def accuracy_func(prbs, labels, total_words):
    # label --> batch * seq_length
    # lengths --> batch
    # prbs -> batch * seq_length * vocab_szie

    total_correct = 0
    pred = torch.argmax(prbs, dim=2)  # batch * seq_length
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] == 0:
                break
            total_correct += 1 if pred[i][j] == labels[i][j] else 0
    acc = total_correct / total_words
    return acc, total_correct, total_words



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--elmo", action="store_true",
                        help="use elmo")
    parser.add_argument("-b", "--bert", action="store_true",
                        help="use bert")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    hyperparams["model_type"] = 'bert' if args.bert else "elmo"
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    embedder_type = 'bert' if args.bert else 'elmo'


    # training_dataset = AGNewsDataset(embedder_type, 'train', hyperparams['window_size'])
    vocab_dict = dict()
    training_dataset = AGNewsDataset(embedder_type, 'train', hyperparams['window_size'], vocab_dict = vocab_dict)
    vocab_dict = training_dataset.vocab_dict
    testing_dataset = AGNewsDataset(embedder_type, 'test', hyperparams['window_size'], vocab_dict = vocab_dict)
    vocab_dict = testing_dataset.vocab_dict
    vocab_size = testing_dataset.vocab_size
    print("final vocab size: ", vocab_size)

    train_loader = DataLoader(training_dataset, batch_size = hyperparams['batch_size'], shuffle = True)
    test_loader = DataLoader(testing_dataset, batch_size = hyperparams['batch_size'], shuffle = True)
    # train_loader = []
    # batch_size = hyperparams['batch_size']
    # start = 0
    # print("preparing training data")
    # while start + batch_size < len(training_dataset):
    #     batch = []
    #     for i in range(start, start + batch_size):
    #         batch.append(training_dataset[i])
    #     train_loader.append(batch)
    #     start += batch_size

    # test_loader = []
    # start = 0
    # print("preparing testing data")
    # while start + batch_size < len(testing_dataset):
    #     batch = []
    #     for i in range(start, start + batch_size):
    #         batch.append(testing_dataset[i])
    #     test_loader.append(batch)
    #     start += batch_size

    # embedder_type = 'bert' if args.bert else 'elmo'

    model = LSTMLM(
        embedder_type,
        hyperparams["rnn_size"],
        vocab_size
    ).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train(model, train_loader, experiment, hyperparams)
    if args.test:
        print("running testing loop...")
        test(model, test_loader, experiment, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')