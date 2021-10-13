from typing import Any, Callable, Dict
from copy import deepcopy

import numpy as np

import torch
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class EngineForNCF(Engine):

    def __init__(self, func: Callable, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch: Dict[str, Any]) -> Dict[str, float]:
        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = mini_batch['targets'], mini_batch['labels']
        x, y = x.to(engine.device), y.to(engine.device)

        y_hat = engine.model(users=x[:, 0], urls=x[:, 1])

        loss = engine.crit(y_hat, y)
        loss.backward()

        if isinstance(y, torch.FloatTensor) or isinstance(y, torch.cuda.FloatTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0
        
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }
    
    @staticmethod
    def validate(engine, mini_batch: Dict[str, Any]) -> Dict[str, float]:
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch['targets'], mini_batch['labels']
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(users=x[:, 0], urls=x[:, 1])

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.FloatTensor) or isinstance(y, torch.cuda.FloatTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0
        
        return {
            'loss': float(loss),
            'accuarcy': float(accuracy)
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )


class NCFTrainer():

    def __init__(self, config) -> None:
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader
    ):
        train_engine = EngineForNCF(
            EngineForNCF.train,
            model, crit, optimizer, self.config
        )
        validation_engine = EngineForNCF(
            EngineForNCF.validate,
            model, crit, optimizer, self.config
        )

        EngineForNCF.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)
        
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine, valid_loader
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            EngineForNCF.check_best
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs
        )

        model.load_state_dict(validation_engine.best_model)

        return model