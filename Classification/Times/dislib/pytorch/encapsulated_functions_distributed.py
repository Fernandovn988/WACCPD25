import numpy as np
import torch

import torch.nn as nn
import copy
from pycompss.api.api import compss_wait_on, compss_barrier, compss_delete_object
from dislib.data.tensor import shuffle
from dislib.pytorch.pytorch_distributed import PytorchDistributed
import time
from sklearn.metrics import accuracy_score
import math

def pt_aggregateParameters(workers_parameters):
    NUM_WORKERS = len(workers_parameters)

    final_weights = []
    for i in range(NUM_WORKERS):
        workers_weights = []
        for param in workers_parameters[i].parameters():
            workers_weights.append(param)
        final_weights.append(workers_weights)
    final_added_parameters = final_weights[0]
    for i in range(len(final_weights[0])):
        for j in range(1, len(final_weights)):
            final_added_parameters[i] = final_added_parameters[i] + \
                                        final_weights[j][i]

    for i in range(len(final_weights[0])):
        final_added_parameters[i] = final_added_parameters[i]/NUM_WORKERS
    j = 0
    if hasattr(workers_parameters[0], 'neural_network_layers'):
        len_nn = len(workers_parameters[0].neural_network_layers)
        for i in range(len_nn):
            if hasattr(workers_parameters[0].neural_network_layers[i],
                       'weight'):
                workers_parameters[0].neural_network_layers[i].weight = \
                    nn.Parameter(final_added_parameters[j])
                j += 1
                workers_parameters[0].neural_network_layers[i].bias = \
                    nn.Parameter(final_added_parameters[j])
                j += 1
            if hasattr(workers_parameters[0].neural_network_layers[i],
                       'shortcut'):
                len_shortcut = len(workers_parameters[0].neural_network_layers[i].shortcut)
                for k in range(len_shortcut):
                    if hasattr(workers_parameters[0].neural_network_layers[i].shortcut[k],
                           'weight'):
                        workers_parameters[0].neural_network_layers[i].shortcut[k].weight = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
                        workers_parameters[0].neural_network_layers[i].shortcut[k].bias = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
                    if hasattr(workers_parameters[0].neural_network_layers[i].shortcut[k],
                           'alpha'):
                        workers_parameters[0].neural_network_layers[i].shortcut[k].alpha = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
            if hasattr(workers_parameters[0].neural_network_layers[i],
                       'layers'):
                len_layers = len(workers_parameters[0].neural_network_layers[i].layers)
                for k in range(len_layers):
                    if hasattr(workers_parameters[0].neural_network_layers[i].layers[k],
                           'weight'):
                        workers_parameters[0].neural_network_layers[i].layers[k].weight = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
                        workers_parameters[0].neural_network_layers[i].layers[k].bias = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
                    if hasattr(workers_parameters[0].neural_network_layers[i].layers[k],
                           'alpha'):
                        workers_parameters[0].neural_network_layers[i].layers[k].alpha = \
                            nn.Parameter(final_added_parameters[j])
                        j += 1
    if hasattr(workers_parameters[0], 'dense_neural_network_layers'):
        len_nn = len(workers_parameters[0].dense_neural_network_layers)
        aux_j = 0
        for i in range(len_nn):
            if hasattr(workers_parameters[0].dense_neural_network_layers[i],
                       'weight'):
                workers_parameters[0].dense_neural_network_layers[i].weight = \
                    nn.Parameter(final_added_parameters[aux_j + j])
                aux_j += 1
                workers_parameters[0].dense_neural_network_layers[i].bias = \
                    nn.Parameter(final_added_parameters[aux_j + j])
                aux_j += 1
    return workers_parameters[0]

def assign_parameters(model, trained_weights):
    j=0
    if hasattr(model, 'neural_network_layers'):
        len_nn = len(model.neural_network_layers)
        for i in range(len_nn):
            if hasattr(model.neural_network_layers[i], 'weight'):
                model.neural_network_layers[i].weight = nn.Parameter(trained_weights.neural_network_layers[i].weight)
                j += 1
                model.neural_network_layers[i].bias = nn.Parameter(trained_weights.neural_network_layers[i].bias)
                j += 1
            if hasattr(model.neural_network_layers[i], 'shortcut'):
                len_shortcut = len(model.neural_network_layers[i].shortcut)
                for k in range(len_shortcut):
                    if hasattr(model.neural_network_layers[i].shortcut[k],
                           'weight'):
                        model.neural_network_layers[i].shortcut[k].weight = \
                            nn.Parameter(trained_weights.neural_network_layers[i].shortcut[k].weight)
                        j += 1
                        model.neural_network_layers[i].shortcut[k].bias = \
                            nn.Parameter(trained_weights.neural_network_layers[i].shortcut[k].bias)
                        j += 1
                    if hasattr(model.neural_network_layers[i].shortcut[k],
                           'alpha'):
                        model.neural_network_layers[i].shortcut[k].alpha = \
                            nn.Parameter(trained_weights.neural_network_layers[i].shortcut[k].alpha)
                        j += 1
            if hasattr(model.neural_network_layers[i],
                       'layers'):
                len_layers = len(model.neural_network_layers[i].layers)
                for k in range(len_layers):
                    if hasattr(model.neural_network_layers[i].layers[k],
                           'weight'):
                        model.neural_network_layers[i].layers[k].weight = \
                            nn.Parameter(trained_weights.neural_network_layers[i].layers[k].weight)
                        j += 1
                        model.neural_network_layers[i].layers[k].bias = \
                            nn.Parameter(trained_weights.neural_network_layers[i].layers[k].bias)
                        j += 1
                    if hasattr(model.neural_network_layers[i].layers[k],
                           'alpha'):
                        model.neural_network_layers[i].layers[k].alpha = \
                            nn.Parameter(trained_weights.neural_network_layers[i].layers[k].alpha)
                        j += 1
    if hasattr(model, 'dense_neural_network_layers'):
        len_nn = len(model.dense_neural_network_layers)
        aux_j = 0
        for i in range(len_nn):
            if hasattr(model.dense_neural_network_layers[i], 'weight'):
                model.dense_neural_network_layers[i].weight = nn.Parameter(trained_weights.dense_neural_network_layers[i].weight)
                aux_j += 1
                model.dense_neural_network_layers[i].bias = nn.Parameter(trained_weights.dense_neural_network_layers[i].bias)
                aux_j += 1
    return model



class EncapsulatedFunctionsDistributedPytorch(object):
    """
    Object that encapsulates the different distributed trainings that can be
    done using PyCOMPSs. Each function implements a different version, the
    number of epochs and batches is specified in each of the functions.

    There are mainly three different types of training.

    - Synchronous training: At the end of each epoch, the weights are
    synchronized and the update is computed.
    - Partially asynchronous: The weights of each worker are updated
    commutatively with the general weights and viceversa.
    - Asynchronous training: A synchronization and update of the weigths is
    done after executing all the epochs or each n specified epochs.

    Attributes
    ----------
    model_parameters : tensor
        weights and biases of the different layers of the network that
        is being trained.
    compss_object: list
        List that contains objects of type PytorchDistributed, each of the
        objects in this list makes a small part of the epoch training in
        parallel to the rest.
    num_workers: int
        Number of parallel trainings existing.

    """
    def __init__(self, num_workers=10):
        self.model_parameters = None
        self.num_workers = num_workers

    def build(self, net, optimizer, loss, optimizer_parameters,
              scheduler=None, T_max=1, eta_min=0.0, num_gpu=0, num_nodes=0):
        """
        Builds the model to obtain the initial parameters of the training
        and it also builds the model in each worker in order to be ready
        to start the training.

        Parameters
        ----------
        net : pytorch Model
            Neural network model to be used during the parallel training.
        optimizer: dict
            Dictionary containing the optimizer to be used and its parameters.
        loss: str
            String specifying the loss to be used during the training.
        num_gpu: int
            Number of GPUs to use during the training.
        num_nodes: int
            Number of nodes available during the training.
        Returns
        -------
        (void)
        """
        if num_gpu > 0:
            self.compss_object = [PytorchDistributed() for _ in
                                  range(num_gpu*num_nodes)]
            for i in range(num_gpu*num_nodes):
                self.compss_object[i].build(copy.deepcopy(net), loss,
                                            copy.deepcopy(optimizer),
                                            optimizer_parameters,
                                            scheduler=scheduler)
        self.optimizer_parameters = optimizer_parameters
        self.num_gpu = num_gpu
        self.num_gpus_per_worker = int(num_nodes*num_gpu/self.num_workers)
        self.model_parameters = net
        self.optimizer = optimizer(self.model_parameters.parameters(),
                **optimizer_parameters)
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, T_max=T_max, eta_min=eta_min)
        else:
            self.scheduler = None

    def get_parameters(self):
        """
        Returns the parameters (weights) of the neural network
        Returns
        -------
        model_parameters: np.array
        """
        return self.model_parameters

    def fit_synchronous_shuffle_every_n_epochs_with_GPU(self, x_train,
                                                        y_train,
                                                        num_batches_per_worker,
                                                        num_epochs,
                                                        n_epocs_sync=1):
        """
        Training of the neural network performing a syncrhonization every n
        specified epochs, it performs a total shuffle of the dataset used.

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every piece
             of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing a syncrhonization and
            between synchronizations
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=False)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                parameters_for_workers = compss_wait_on(
                    parameters_for_workers)
                self.model_parameters = \
                    pt_aggregateParameters(parameters_for_workers)
                parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                          for _ in
                                          range(len(parameters_for_workers))]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregateParameters(
            parameters_for_workers)
        return self.model_parameters

    def fit_synchronous_every_n_epochs_with_GPU(self, x_train, y_train,
                                                num_batches_per_worker,
                                                num_epochs,
                                                n_epocs_sync=1,
                                                shuffle_blocks=True,
                                                shuffle_block_data=True):
        """
        Training of the neural network performing a syncrhonization every n
        specified epochs,  it performs a total shuffle of the tensors on the
        ds_tensor and the elements inside each tensor

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every
            piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing a syncrhonization
            and between synchronizations
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside each
            tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        for i in range(num_epochs):
            j = 0
            if shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            for row in rows:
                for col in cols:
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                parameters_for_workers = compss_wait_on(
                    parameters_for_workers)
                self.model_parameters = \
                    pt_aggregateParameters(parameters_for_workers)
                parameters_for_workers = [
                    copy.deepcopy(self.model_parameters) for _
                    in range(len(parameters_for_workers))]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregateParameters(
            parameters_for_workers)
        return self.model_parameters
    
    def fit_synchronous_with_GPU(self, x_train, y_train,
                                 num_batches_per_worker,
                                 num_epochs,
                                 shuffle_blocks=True,
                                 shuffle_block_data=True,
                                 make_validation=True, return_loss=False):
        """
        Training of the neural network performing a syncrhonization of the
        weights at the end of each epoch, it performs a total shuffle of
        the tensors on the ds_tensor and the elements inside each tensor

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every
            piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside each
            tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters) for
                                  _ in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        #time_per_epoch = []
        #total_time = 0
        for i in range(num_epochs):
            j = 0
            if shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            epoch_accuracy = []
            epoch_loss = []
            for row in rows:
                for col in cols:
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if self.scheduler is not None:
                self.scheduler.step()
                self.optimizer_parameters = {}
                self.optimizer_parameters["lr"] = self.optimizer.param_groups[0]["lr"]
                if "momentum" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["momentum"] = self.optimizer.param_groups[0]["momentum"]
                if "weight_decay" in self.optimizer.param_groups[0]:
                    self.optimizer_parameters["weight_decay"] = self.optimizer.param_groups[0]["weight_decay"]
            parameters_for_workers = compss_wait_on(parameters_for_workers)
            #start_time = time.time()
            self.model_parameters = \
                pt_aggregateParameters(parameters_for_workers)
            #actual_time = time.time() - start_time
            #total_time += actual_time
            #time_per_epoch.append(actual_time)
            parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                      for _ in range(self.num_workers)]
        self.model_parameters = parameters_for_workers[0]
        #print("TIME PER EPOCH")
        #print(total_time/num_epochs)
        #print(time_per_epoch)
        return parameters_for_workers[0]

    def fit_synchronous_shuffle_with_GPU(self, x_train, y_train,
                                         num_batches_per_worker,
                                         num_epochs):
        """
        Training of the neural network performing a syncrhonization of
        the weights every epoch, it performs a total shuffle of the dataset

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with
            every piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=False)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            parameters_for_workers = compss_wait_on(parameters_for_workers)
            self.model_parameters = \
                pt_aggregateParameters(parameters_for_workers)
            parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                      for _ in range(self.num_workers)]
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregateParameters(parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_with_GPU(self, x_train, y_train,
                                  num_batches_per_worker,
                                  num_epochs,
                                  shuffle_blocks=True,
                                  shuffle_block_data=True):
        """
        Training of the neural network performing an asyncrhonous update
        of the weights every epoch, it performs a shuffle of the tensors
        on the ds_tensor and a local shuffle of the elements inside each
        tensor

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every
            piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside each
            tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        j = 0
        for i in range(num_epochs):
            if shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            for row in rows:
                for col in cols:
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            for j in range(self.num_workers):
                parameters_for_workers[j] = \
                    self.compss_object[j].aggregate_parameters_async(
                        self.model_parameters,
                        parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregateParameters(parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_shuffle_with_GPU(self, x_train, y_train,
                                          num_batches_per_worker,
                                          num_epochs):
        """
        Training of the neural network performing an asyncrhonous
        update of the weights every epoch, it performs a total shuffle
         of the dataset

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with
            every piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=False)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            for j in range(self.num_workers):
                parameters_for_workers[j] = \
                    self.compss_object[j].aggregate_parameters_async(
                        self.model_parameters,
                        parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregateParameters(parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_n_epochs_with_GPU(self, x_train, y_train,
                                           num_batches_per_worker,
                                           num_epochs,
                                           n_epocs_sync=0,
                                           shuffle_blocks=True,
                                           shuffle_block_data=True):
        """
        Training of the neural network performing an asyncrhonous update
        of the weights every n epochs, it performs a shuffle of the tensors
        and locally a shuffle of the elements inside each tensor

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every
            piece of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing an asyncrhonous
            update of the weights and between the following updates
        shuffle_blocks: boolean
            Variable specifying to shuffle the blocks of the ds_tensor or not
        shuffle_block_data: boolean
            Variable specifying whether to shuffle the elements inside each
            tensor locally or not
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        rows = np.arange(x_train.shape[0])
        cols = np.arange(x_train.shape[1])
        for i in range(num_epochs):
            j = 0
            if shuffle_blocks:
                rows = np.random.permutation(x_train.shape[0])
                cols = np.random.permutation(x_train.shape[1])
            for row in rows:
                for col in cols:
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=shuffle_block_data)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                for j in range(self.num_workers):
                    parameters_for_workers[j] = \
                        self.compss_object[j].aggregate_parameters_async(
                            self.model_parameters,
                            parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregateParameters(parameters_for_workers)
        return self.model_parameters

    def fit_asynchronous_shuffle_n_epochs_with_GPU(self, x_train,
                                                   y_train,
                                                   num_batches_per_worker,
                                                   num_epochs,
                                                   n_epocs_sync=0):
        """
        Training of the neural network performing an asyncrhonous update
        of the weights every n epochs, it performs a total shuffle of the
        dataset

        Parameters
        ----------
        x_train : ds_tensor
            samples and features of the training dataset
        y_train: ds_tensor
            classes or values of the samples of the training dataset
        num_batches_per_worker: int
            Number of batches that each worker will be trained with every piece
            of the dataset
        num_epochs: int
            Total number of epochs to train the model
        n_epocs_sync: int
            Number of epochs to train before performing an asyncrhonous update
            of the weights and between the following updates
        Returns
        -------
        model_parameters: np.array
        """
        parameters_for_workers = [copy.deepcopy(self.model_parameters)
                                  for _ in range(self.num_workers)]
        pt_aggregateParameters(parameters_for_workers)
        for i in range(num_epochs):
            j = 0
            x_train, y_train = shuffle(x_train, y_train)
            rows = x_train.shape[0]
            cols = x_train.shape[1]
            for row in range(rows):
                for col in range(cols):
                    parameters_for_workers[j] = \
                        self.compss_object[j].train_cnn_batch_GPU(
                            parameters_for_workers[j],
                            x_train.tensors[int(row)][int(col)],
                            y_train.tensors[int(row)][int(col)],
                            num_batches_per_worker,
                            shuffle_block_data=False)
                    j = j + 1
                    if j == self.num_workers:
                        j = 0
            if (i + 1) % n_epocs_sync == 0:
                for j in range(self.num_workers):
                    parameters_for_workers[j] = \
                        self.compss_object[j].aggregate_parameters_async(
                            self.model_parameters,
                            parameters_for_workers[j])
        parameters_for_workers = compss_wait_on(parameters_for_workers)
        self.model_parameters = pt_aggregateParameters(parameters_for_workers)
        return self.model_parameters


def process_outputs(output_nn):
    _, indices = torch.max(output_nn, dim=1)
    binary_output = torch.zeros_like(output_nn)
    binary_output[torch.arange(output_nn.size(0)), indices] = 1
    return binary_output
