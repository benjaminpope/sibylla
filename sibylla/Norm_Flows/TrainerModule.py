"""
Trainer module for normalising flows
Taken from
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial11/NF_image_modeling.html
"""


import os
import optax
import jax
import jax.numpy as np
from tqdm import tqdm
import time
import json

from torch.utils.tensorboard import SummaryWriter
from flax.training import train_state, checkpoints

from DataLoader import DataLoader


class TrainerModule:
    def __init__(self, model_name, train_exmp_loader, train_data_loader, checkpoint_path, flow, lr=1e-3, seed=42):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = flow
        self.checkpoint_path = checkpoint_path
        # Prepare logging
        self.train_exmp_loader = train_exmp_loader
        self.train_data_loader = train_data_loader
        self.exmp_imgs = next(iter(self.train_exmp_loader))[0]
        self.log_dir = os.path.join(checkpoint_path, self.model_name)
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()

    def create_functions(self):
        # Training function
        def train_step(state, rng, batch):
            imgs, _ = batch
            loss_fn = lambda params: self.model.apply({'params': params}, imgs, rng, testing=False)
            # Get loss and gradients for loss
            (loss, rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            return state, rng, loss
        self.train_step = jax.jit(train_step)

        # Eval function, which is separately jitted for validation and testing
        def eval_step(state, rng, batch, testing):
            return self.model.apply({'params': state.params}, batch[0], rng, testing=testing)
        self.eval_step = jax.jit(eval_step, static_argnums=(3,))

    def init_model(self):
        # Initialize model
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng, init_rng, flow_rng = jax.random.split(self.rng, 3)
        params = self.model.init(init_rng, self.exmp_imgs, flow_rng)['params']
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.exponential_decay(
            init_value=self.lr,
            transition_steps=len(self.train_data_loader),
            decay_rate=0.99,
            end_value=0.01 * self.lr
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at 1
            optax.adam(lr_schedule)
        )
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)

    def train_model(self, train_loader, val_loader, num_epochs=500):
        # Train model for defined number of epochs
        best_eval = 1e6
        # for epoch_idx in tqdm(range(1, num_epochs + 1)):
        for epoch_idx in range(1, num_epochs + 1):
            print(f"Epoch {epoch_idx}")
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 5 == 0:
                eval_bpd = self.eval_model(val_loader, testing=False)
                self.logger.add_scalar('val/bpd', eval_bpd, global_step=epoch_idx)
                if eval_bpd < best_eval:
                    best_eval = eval_bpd
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, data_loader, epoch):
        # Train model for one epoch, and log avg loss
        avg_loss = 0.
        # for batch in tqdm(data_loader, leave=False):
        for batch_idx, batch in enumerate(data_loader):
            print(f"\t batch_idx: {batch_idx}")
            self.state, self.rng, loss = self.train_step(self.state, self.rng, batch)
            avg_loss += loss
        avg_loss /= len(data_loader)
        self.logger.add_scalar('train/bpd', avg_loss.item(), global_step=epoch)

    def eval_model(self, data_loader, testing=False):
        # Test model on all images of a data loader and return avg loss
        losses = []
        batch_sizes = []
        for batch in data_loader:
            loss, self.rng = self.eval_step(self.state, self.rng, batch, testing=testing)
            losses.append(loss)
            batch_sizes.append(batch[0].shape[0])
        losses_np = np.stack(jax.device_get(losses))
        batch_sizes_np = np.stack(batch_sizes)
        avg_loss = (losses_np * batch_sizes_np).sum() / batch_sizes_np.sum()
        return avg_loss

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, step=step)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.checkpoint_path,
                                                    f'{self.model_name}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(self.checkpoint_path, f'{self.model_name}.ckpt'))

    def train_flow(self):
        # TODO: make this a class method
        # Create a trainer module with specified hyperparameters
        if self.model_name[0:9] == "MNISTFlow":
            train_set, val_set, test_set = DataLoader.load_MNIST()
            train_exmp_loader, train_data_loader, \
                val_loader, test_loader = DataLoader.generate_data_loaders(train_set,
                                                                           val_set,
                                                                           test_set)
        else:
            raise NotImplementedError()

        if not self.checkpoint_exists():  # Skip training if pretrained model exists
            self.train_model(train_data_loader,
                             val_loader,
                             num_epochs=200)
            self.load_model()
            val_bpd = self.eval_model(val_loader, testing=True)
            start_time = time.time()
            test_bpd = self.eval_model(test_loader, testing=True)
            duration = time.time() - start_time
            results = {'val': val_bpd,
                       'test': test_bpd,
                       'time': duration / len(test_loader) / self.model.import_samples}
        else:
            self.load_model(pretrained=True)
            with open(os.path.join(self.checkpoint_path, f'{self.model_name}_results.json'), 'r') as f:
                results = json.load(f)

        # Bind parameters to model for easier inference
        self.model_bd = self.model.bind({'params': self.state.params})
        return self, results
