import os
from logging import getLogger

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from GiantTourEnv import GiantTourEnv
from GiantTourModel import GiantTourModel
from utils.utils import AverageMeter, LogData, TimeEstimator, get_result_folder


class GiantTourTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.logger = getLogger("trainer")
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        use_cuda = trainer_params["use_cuda"] and torch.cuda.is_available()
        if trainer_params["use_cuda"] and not use_cuda:
            self.logger.warning("CUDA was requested but is unavailable; falling back to CPU")
        self.device = torch.device(
            "cuda", trainer_params["cuda_device_num"]
        ) if use_cuda else torch.device("cpu")
        if use_cuda:
            torch.cuda.set_device(self.device)

        self.model = GiantTourModel(**model_params).to(self.device)
        env_params_with_device = dict(env_params)
        env_params_with_device["device"] = self.device
        self.env = GiantTourEnv(**env_params_with_device)
        self.optimizer = Adam(self.model.parameters(), **optimizer_params["optimizer"])
        self.scheduler = MultiStepLR(self.optimizer, **optimizer_params["scheduler"])

        self.start_epoch = 1
        model_load = trainer_params.get("model_load", {"enable": False})
        if model_load.get("enable", False):
            checkpoint_name = os.path.join(
                model_load["path"], "checkpoint-{}.pt".format(model_load["epoch"])
            )
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "result_log" in checkpoint:
                self.result_log.set_raw_data(checkpoint["result_log"])
            self.start_epoch = checkpoint["epoch"] + 1

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        epochs = self.trainer_params["epochs"]
        for epoch in range(self.start_epoch, epochs + 1):
            score, loss = self._train_one_epoch(epoch)
            self.result_log.append("train_score", epoch, score)
            self.result_log.append("train_loss", epoch, loss)
            self.scheduler.step()

            elapsed, remaining = self.time_estimator.get_est_string(epoch, epochs)
            self.logger.info(
                "Epoch %d/%d: tour_score=%.6f loss=%.6f elapsed=%s remaining=%s",
                epoch, epochs, score, loss, elapsed, remaining,
            )

            save_interval = self.trainer_params["logging"]["model_save_interval"]
            if epoch == epochs or epoch % save_interval == 0:
                self._save_checkpoint(epoch)

    def _train_one_epoch(self, epoch):
        score_meter = AverageMeter()
        loss_meter = AverageMeter()
        episode = 0
        total = self.trainer_params["train_episodes"]
        while episode < total:
            batch_size = min(self.trainer_params["train_batch_size"], total - episode)
            score, loss = self._train_one_batch(batch_size)
            score_meter.update(score, batch_size)
            loss_meter.update(loss, batch_size)
            episode += batch_size
        return score_meter.avg, loss_meter.avg

    def _train_one_batch(self, batch_size):
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.empty(
            batch_size, self.env.pomo_size, 0, device=self.device
        )
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        advantage = reward - reward.mean(dim=1, keepdim=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = (-advantage * log_prob).mean()

        best_reward = reward.max(dim=1).values
        score = -best_reward.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return score.item(), loss.item()

    def _save_checkpoint(self, epoch):
        os.makedirs(self.result_folder, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "result_log": self.result_log.get_raw_data(),
                "env_params": self.env_params,
                "model_params": self.model_params,
            },
            os.path.join(self.result_folder, "checkpoint-{}.pt".format(epoch)),
        )
