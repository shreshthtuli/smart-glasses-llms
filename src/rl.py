from src.selector.templateselector import TemplateSelector

from torchrl.envs import EnvBase
from torchrl.data import OneHotDiscreteTensorSpec, UnboundedContinuousTensorSpec, \
    CompositeSpec, BoundedTensorSpec
from torchrl.modules import EGreedyModule, MLP, QValueModule, ProbabilisticActor, \
    TanhNormal, ValueOperator
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import DQNLoss, SoftUpdate, ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torch.optim import Adam
from torch import nn

from rich.progress import track
from transformers import AutoModel
import pandas as pd
import numpy as np
import torch

class SmartGlassesEnvironment(EnvBase):
    def __init__(self, DATA_PATH, model: str = 'jinaai/jina-embeddings-v2-base-en'):
        super().__init__()
        # load data
        selector = TemplateSelector(DATA_PATH)
        self.train_dset = selector.train_dset
        self.test_dset = selector.test_dset
        self.all_llm_names = selector.all_llm_names
        self.transformer = AutoModel.from_pretrained(model, trust_remote_code=True)
        # add embeddings
        for dset in [self.train_dset, self.test_dset]:
            dset['embedding'] = dset.turns.apply(lambda x: self.transformer.encode(x[0]))
        self.max_time = max([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        self.min_time = min([self.train_dset[llm_name+'_time'].median() for llm_name in self.all_llm_names])
        # set properties
        self.dtype = np.float32
        self.state_size = self.transformer.encode('hi').size
        self.action_size = len(self.all_llm_names)
        self.state = self.train_dset.embedding.sample().to_numpy()[0]
        self.action_spec = OneHotDiscreteTensorSpec(self.action_size)
        observation_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.state_size])) 
        self.observation_spec = CompositeSpec(observation=observation_spec)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]))

    def _reset(self, tensordict, **kwargs):
        out_tensordict = TensorDict({}, batch_size=torch.Size())
        self.state = self.train_dset.embedding.sample().to_numpy()[0]
        out_tensordict.set("observation", torch.tensor(self.state.flatten()))
        return out_tensordict

    def _step(self, tensordict):
        action = tensordict["action"]
        action = np.argmax(action.cpu().numpy())
        next_state = torch.tensor(self.train_dset.embedding.sample().to_numpy()[0], device='cpu')
        selection = self.all_llm_names[action]
        row_id = np.where(np.all(np.vstack(self.train_dset.embedding.values) == self.state[np.newaxis, :], axis=1))[0][0]
        row = self.train_dset.iloc[row_id]
        time = row[selection+'_time'] 
        score = row[selection+'_score']
        reward = row['complexity'] * (score-1)/9 + \
            row['time_criticality'] * (time-self.min_time)/(self.max_time-self.min_time)
        reward = np.array(reward)
        out_tensordict = TensorDict({"observation": next_state,
                                     "reward": torch.tensor(reward.astype(np.float32)),
                                     "done": True}, batch_size=torch.Size())
        return out_tensordict

    def _set_seed(self, seed):
        pass

class DQNPolicy():
    def __init__(self, env:SmartGlassesEnvironment, SAVE_PATH='./models/'):
        self.env = env
        value_mlp = MLP(out_features=self.env.action_spec.shape[-1], num_cells=[64, 64])
        value_net = TensorDictModule(value_mlp, in_keys=["observation"], out_keys=["action_value"])
        self.policy = TensorDictSequential(value_net, QValueModule(self.env.action_spec))
        self.exploration_module = EGreedyModule(
            self.env.action_spec, annealing_num_steps=100_000, eps_init=0.5
        )
        self.initialize_collector()
        self.loss = DQNLoss(value_network=self.policy, 
                       action_space=self.env.action_spec, 
                       delay_value=True)
        self.updater = SoftUpdate(self.loss, eps=0.99)
        self.initialize_optimizer()

    def initialize_collector(self):
        self.init_rand_steps = 5000
        self.optim_steps = 10
        self.collector = SyncDataCollector(
            self.env,
            self.policy,
            frames_per_batch=1,
            total_frames=2e4,
            split_trajs=False,
        )
        self.rb = ReplayBuffer(storage=LazyTensorStorage(100_000),
                               sampler=SamplerWithoutReplacement())
    
    def initialize_optimizer(self):
        self.optim = Adam(self.loss.parameters(), lr=0.002)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.collector.total_frames, 0.0
        )
        self.logger = TensorboardLogger(log_dir='logs', exp_name=self.__class__.__name__)

    def train(self):
        total_count = 0
        total_episodes = 0
        for i, data in track(enumerate(self.collector), total=self.collector.total_frames):
            try:
                self.advantage(data)
            except:
                pass
            # Write data in replay buffer
            self.rb.extend(data)
            if len(self.rb) > self.init_rand_steps:
                # Optim loop (we do several optim steps
                # per batch collected for efficiency)
                for _ in range(self.optim_steps):
                    sample = self.rb.sample(128)
                    loss_vals = self.loss(sample)
                    loss = 0
                    for key in [item[0] for item in loss_vals.items()]:
                        if "loss" in key:
                            loss += loss_vals[key]
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    try:
                        # Update exploration factor
                        self.exploration_module.step(data.numel())
                        # Update target params
                        self.updater.step()
                    except:
                        pass
                    total_count += data.numel()
                    total_episodes += data["next", "done"].sum()
            if i % 10 == 0:
                with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                    eval_rollout = self.env.rollout(1000, self.policy)
                    self.logger.log_scalar("eval reward", eval_rollout["next", "reward"].mean().item(), i)
                    self.logger.log_scalar("eval reward (sum)", 
                        eval_rollout["next", "reward"].sum().item(), i
                    )
                    self.logger.log_scalar("eval step_count", eval_rollout["step_count"].max().item(), i)
                    del eval_rollout
            self.scheduler.step()

class PPOPolicy(DQNPolicy):
    def __init__(self, env:SmartGlassesEnvironment):
        self.env = env
        actor_net = nn.Sequential(
            nn.LazyLinear(128),
            nn.Tanh(),
            nn.LazyLinear(128),
            nn.Tanh(),
            nn.LazyLinear(128),
            nn.Tanh(),
            nn.LazyLinear(2 * self.env.action_spec.shape[-1]),
            NormalParamExtractor(),
        )
        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True
        )
        value_net = nn.Sequential(
            nn.LazyLinear(128),
            nn.Tanh(),
            nn.LazyLinear(128),
            nn.Tanh(),
            nn.LazyLinear(128),
            nn.Tanh(),
            nn.LazyLinear(1),
        )
        self.value = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )
        self.initialize_collector()
        self.policy(env.reset())
        self.value(env.reset())
        self.advantage = GAE(gamma=0.99, lmbda=0.95, value_network=self.value, average_gae=True)
        self.loss = ClipPPOLoss(actor_network=self.policy,
                                critic_network=self.value,
                                entropy_bonus=bool(1e-4),
                                entropy_coef=1e-4,
                                critic_coef=1.0,
                                loss_critic_type='smooth_l1'
                                )
        self.initialize_optimizer()
        self.policy