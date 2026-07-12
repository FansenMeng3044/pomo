import torch
from logging import getLogger

from GiantTourEnv import GiantTourEnv as Env
from GiantTourModel import GiantTourModel as Model

from utils.utils import AverageMeter, TimeEstimator, get_result_folder


class GiantTourTester:
    """Evaluate a giant-tour policy by SPLITTING its output into CVRP routes.

    The policy is trained with the raw giant-tour reward (no split), but here
    the environment is created with ``split_reward=True`` so the reported score
    is the feasible, hard-capacity CVRP distance produced by optimal Split.
    Metrics (no-aug and x8-aug) match the official CVRPTester definition.
    """

    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger(name="tester")
        self.result_folder = get_result_folder()

        use_cuda = tester_params["use_cuda"] and torch.cuda.is_available()
        if use_cuda:
            cuda_device_num = tester_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")
        self.device = device

        # Evaluation always scores the split (feasible CVRP) cost.
        env_params_eval = dict(env_params)
        env_params_eval["device"] = device
        env_params_eval["split_reward"] = True
        self.env = Env(**env_params_eval)
        self.model = Model(**model_params)

        model_load = tester_params["model_load"]
        checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if self.tester_params["test_data_load"]["enable"]:
            self.env.use_saved_problems(
                self.tester_params["test_data_load"]["filename"], self.device
            )

        test_num_episode = self.tester_params["test_episodes"]
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params["test_batch_size"], remaining)

            score, aug_score = self._test_one_batch(batch_size)
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                episode, test_num_episode
            )
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], "
                "split_score:{:.4f}, aug_split_score:{:.4f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str,
                    score, aug_score,
                )
            )

            if episode == test_num_episode:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SPLIT SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SPLIT SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size):
        if self.tester_params["augmentation_enable"]:
            aug_factor = self.tester_params["aug_factor"]
        else:
            aug_factor = 1

        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # best over pomo
        no_aug_score = -max_pomo_reward[0, :].float().mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # best over augmentation
        aug_score = -max_aug_pomo_reward.float().mean()

        return no_aug_score.item(), aug_score.item()
