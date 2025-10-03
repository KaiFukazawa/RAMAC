import argparse
import gym
import numpy as np
import os
import torch
import json
import yaml

import d4rl
from utils.data_sampler import Data_Sampler
from utils.experiment import print_banner, EarlyStopping
from utils.logger import logger, setup_logger
import h5py
from torch.utils.tensorboard import SummaryWriter

# Optional: wrapper for creating risky environments
try:
    from environment.risky_wrappers import make_risky_env
    HAS_RISKY_WRAPPERS = True
    print_banner("Risky Wrappers loaded")
except ImportError:
    HAS_RISKY_WRAPPERS = False
    print_banner("Risky Wrappers not found. Skipping risky env evaluation.")
    pass

def load_hyperparameters(env_name: str, config_path: str):
    """
    Load hyperparameters from YAML configuration file for a specific environment.

    Args:
        env_name (str): Name of the environment to load hyperparameters for
        config_path (str): Path to the YAML configuration file (defaults to env var CONFIG_PATH)

    Returns:
        dict: Dictionary of hyperparameters for the specified environment
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract environment-specific hyperparameters
    environments = config.get('environments', {})
    defaults = config.get('defaults', {})

    if env_name not in environments:
        raise ValueError(f"Environment '{env_name}' not found in configuration file: {config_path}")

    # Get hyperparameters for the specific environment
    hyperparameters = environments[env_name].copy()

    # Fill in any missing parameters with defaults
    for param, default_value in defaults.items():
        if param not in hyperparameters:
            hyperparameters[param] = default_value

    return hyperparameters

def load_dataset(path):
    """
    Return a dict containing observations, actions, etc.
    If the file has a nested 'data' group, descend one level into it.
    """
    if path.endswith('.hdf5'):
        with h5py.File(path, 'r') as hf:
            root = hf['data'] if 'data' in hf else hf
            return {k: root[k][:] for k in root.keys()}
    elif path.endswith('.npy'):
        arr = np.load(path, allow_pickle=True).item()
        return arr['data'] if 'data' in arr else arr      
    else:
        raise ValueError("Unknown dataset format")

def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    """Train the selected agent and periodically evaluate and save artifacts.

    Args:
        env (gym.Env): Initialized environment instance.
        state_dim (int): Observation dimensionality.
        action_dim (int): Action dimensionality.
        max_action (float): Maximum absolute action value.
        device (str): Torch device string, e.g., "cpu" or "cuda:0".
        output_dir (str): Directory path to write logs and artifacts.
        args (argparse.Namespace): Parsed configuration and hyperparameters.

    Returns:
        None
    """
    # ------------------------------------------------
    # 1. Load replay buffer
    # ------------------------------------------------
    if args.risky_dataset_path and os.path.isfile(args.risky_dataset_path):
        if args.risky_dataset_path.endswith('.hdf5'):
            data_dict = load_dataset(args.risky_dataset_path)
            data_sampler = Data_Sampler(data_dict, device=device, reward_tune='no')
            print_banner('Loaded RISKY dataset from .hdf5 file')
            state_arr = data_sampler.state
            if state_arr is None:
                raise RuntimeError("state array not found")
            state_dim = state_arr.shape[1]      # Overwrite env-provided value

            action_arr = data_sampler.action
            if action_arr is None:
                raise RuntimeError("action array not found")
            action_dim = action_arr.shape[1]

        elif args.risky_dataset_path.endswith('.npy'):
            print("Yes, the file exists!")
            loaded_data = np.load(args.risky_dataset_path, allow_pickle=True) #.item()
            print("Loading .npy file now...")
        # For example, including { 'observations', 'actions', 'rewards', 'next_observations', 'terminals' }
            data_sampler = Data_Sampler(loaded_data, device=device, reward_tune='no')
            print_banner(f'Loaded RISKY dataset from {args.risky_dataset_path} (npy)')
        else:
            raise ValueError("Unknown dataset format. Please provide .hdf5 or .npy")

    else:
         # Use standard D4RL buffer
        dataset = d4rl.qlearning_dataset(env)
        data_sampler = Data_Sampler(dataset, device=device)
        print_banner('Loaded standard D4RL buffer')


    # ------------------------------------------------
    # 2. Algorithm selection (algo == 'radac' or 'rafmac')
    # ------------------------------------------------
    if args.algo == 'radac':
        from agents.radac import RADAC
        agent = RADAC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=args.discount,
            tau=args.tau,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.grad_norm,
            n_quantiles=args.n_quantiles,
            embedding_dim=args.emb_dim,
            risk_distortion=args.risk_distortion,
            alpha_cvar=args.alpha,
            ema_decay=0.995,
            eta=args.eta,
            step_start_ema=args.start_steps,
            update_ema_every=5,
            q_clip_range=args.q_clip_range,
            lambda_bc=args.lambda_bc,
            eta_warmup_steps=args.eta_warmup_steps,
            eta_ramp_steps=args.eta_ramp_steps
        )


    elif args.algo == 'rafmac':
        from agents.rafmac import RAFMAC
        agent = RAFMAC(state_dim=state_dim,
                    action_dim=action_dim,
                    max_action=max_action,
                    device=device,
                    discount=args.discount,
                    tau=args.tau,
                    flow_steps=args.flow_steps,        # Set to 1 for 1-step
                    eta=args.eta,
                    hidden_dim=args.emb_dim,
                    normalize_q_loss=args.normalize_q_loss,
                    grad_norm=args.grad_norm,
                    risk_dist=args.risk_distortion,
                    q_agg=args.q_agg,
                    use_distillation=args.use_distillation)

    else:
        raise NotImplementedError(f"Unknown algo: {args.algo}")

    # ------------------------------------------------
    # 3. Setup EarlyStopping and TensorBoard
    # ------------------------------------------------
    early_stop = False
    stop_check = EarlyStopping(tolerance=1, min_delta=0.0)
    writer = SummaryWriter(output_dir) if args.record_tensorboard else None

    # ------------------------------------------------
    # 4. Main training loop
    # ------------------------------------------------
    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.
    offline_checkpoints = []

    print_banner("Training Start", separator="*", num_star=90)

    while (training_iters < max_timesteps):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)

        # 4.1 Training step
        loss_metric = agent.train(
            replay_buffer=data_sampler,
            iterations=iterations,
            batch_size=args.batch_size,
            log_writer=writer
        )

        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Helper function to safely convert tensor or list of tensors to numpy
        def safe_to_numpy(data):
            if isinstance(data, list):
                # If it's a list, convert each element to numpy
                return [item.detach().cpu().numpy() if hasattr(item, 'cpu') else item for item in data]
            elif hasattr(data, 'cpu'):
                # If it's a tensor, convert to numpy
                return data.detach().cpu().numpy()
            else:
                # If it's already a numpy array or scalar
                return data

        bc_loss_avg = np.mean(safe_to_numpy(loss_metric['bc_loss']))
        actor_loss_avg = np.mean(safe_to_numpy(loss_metric['actor_loss']))
        critic_loss_avg = np.mean(safe_to_numpy(loss_metric['critic_loss']))

        if 'cvar_val' in loss_metric:
            cvar_avg = np.mean(safe_to_numpy(loss_metric['cvar_val']))
        else:
            cvar_avg = 0.0

        if 'Q_mean' in loss_metric:
            q_mean_avg = np.mean(safe_to_numpy(loss_metric['Q_mean']))
        else:
            q_mean_avg = 0.0
        if 'IQN_loss' in loss_metric:
            iqn_loss_avg = np.mean(safe_to_numpy(loss_metric['IQN_loss']))
        else:
            iqn_loss_avg = 0.0

        # RAFMAC-specific debug info
        if 'q_abs_mean' in loss_metric:
            q_abs_mean_avg = np.mean(safe_to_numpy(loss_metric['q_abs_mean']))
            td_target_mean_avg = np.mean(safe_to_numpy(loss_metric['td_target_mean']))
            a_student_max_avg = np.mean(safe_to_numpy(loss_metric['a_student_max']))
            not_done_mean_avg = np.mean(safe_to_numpy(loss_metric['not_done_mean']))
        else:
            q_abs_mean_avg = 0.0
            td_target_mean_avg = 0.0
            a_student_max_avg = 0.0
            not_done_mean_avg = 0.0

        print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', bc_loss_avg)
        logger.record_tabular('CVaR', cvar_avg)
        logger.record_tabular('Actor Loss', actor_loss_avg)
        logger.record_tabular('Critic Loss', critic_loss_avg)
        logger.record_tabular('Q Mean', q_mean_avg)
        logger.record_tabular('IQN Loss', iqn_loss_avg)

        # RAFMAC debug info
        if args.algo == 'rafmac':
            logger.record_tabular('Q Abs Mean', q_abs_mean_avg)
            logger.record_tabular('TD Target Mean', td_target_mean_avg)
            logger.record_tabular('A Student Max', a_student_max_avg)
            logger.record_tabular('Not Done Mean', not_done_mean_avg)
        logger.dump_tabular()

        # 4.2 Evaluation
        if args.eval_risky_env and HAS_RISKY_WRAPPERS:
            eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, violation_counts, ep_scores, monitor_vals, monitor_name,algo_name, dataset_name = eval_policy_risky(
                agent, args.env_name, args.seed, args.eval_episodes,
                args.risk_prob, args.risk_penalty,
                args.algo, args.env_name,
                args.max_vel,args.prob_vel_penal, args.cost_vel,
                args.prob_pose_penal, args.cost_pose, args.healthy_angle_range, args.done_if_exceed_factor
            )
            logger.record_tabular('Violation Counts', violation_counts)
            np.save(os.path.join(output_dir, f"ep_returns.npy"), ep_scores)
            if monitor_name is not None:
                file_name = f"{monitor_name}_{algo_name}_{dataset_name}.npy"
                np.save(os.path.join(output_dir, file_name), monitor_vals)
        else:
            eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(
                agent, args.env_name, args.seed, eval_episodes=args.eval_episodes
            )
            violation_counts=0.0

        # RAFMAC-specific evaluations
        if args.algo == 'rafmac':
            evaluations.append([
                eval_res, eval_res_std,
                eval_norm_res, eval_norm_res_std,
                bc_loss_avg, cvar_avg,
                actor_loss_avg, critic_loss_avg,
                curr_epoch, q_mean_avg,
                violation_counts,
                iqn_loss_avg,
                q_abs_mean_avg,
                td_target_mean_avg,
                a_student_max_avg,
                not_done_mean_avg
            ])
        else:
            evaluations.append([
                eval_res, eval_res_std,
                eval_norm_res, eval_norm_res_std,
                bc_loss_avg, cvar_avg,
                actor_loss_avg, critic_loss_avg,
                curr_epoch, q_mean_avg,
                violation_counts,
                iqn_loss_avg
            ])
        np.save(os.path.join(output_dir, "eval"), evaluations)

        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.dump_tabular()

        # 4.4 Model selection (keep top-k models with smallest BC loss)
        if args.save_best_model and args.top_k >= 0:

            actual_k = args.top_k + 1

            # Helper to save the current best model and logs
            def save_best_model_and_logs():
                file_name_best=f"{args.env_name}_{args.exp}_{args.algo}-{args.top_k}"
                best_save_dir = os.path.join("saved_best_models",file_name_best,f"seed{args.seed}")
                os.makedirs(best_save_dir,exist_ok=True)
                print(f"Saving top k model to {best_save_dir} ...")
                agent.save_model(best_save_dir)

                # ---- Additionally save 10-episode returns for the best model ----
                if HAS_RISKY_WRAPPERS and args.eval_risky_env:
                    # Save 10 raw returns on the risky environment
                    np.save(os.path.join(best_save_dir, "ep_returns_raw.npy"),
                            np.array(ep_scores, dtype=np.float32))

                    # Save 10 normalized scores (normalized using the base env)
                    try:
                        base_env = gym.make(args.env_name)
                        norm_scores = [base_env.get_normalized_score(s) for s in ep_scores]
                        np.save(os.path.join(best_save_dir, "ep_returns_norm.npy"),
                                np.array(norm_scores, dtype=np.float32))
                    except Exception as e:
                        print(f"Warning: Could not normalize scores: {e}")
                        # If normalization fails, fall back to raw scores
                        np.save(os.path.join(best_save_dir, "ep_returns_norm.npy"),
                                np.array(ep_scores, dtype=np.float32))

                    # Save metadata
                    eval_meta = {
                        "env": args.env_name,
                        "algo": args.algo,
                        "seed": args.seed,
                        "episodes": int(args.eval_episodes),
                        "selection_rule": "offline_best",
                        "selected_epoch": int(curr_epoch),
                        "use_risky_env": bool(args.eval_risky_env),
                        "alpha": float(args.alpha) if hasattr(args, 'alpha') else None,
                        "metric_pipeline": "per_seed_then_mean",
                        "quantile_method": "linear",
                        "rounding": 1,
                        "normalization_basis": "base_env.get_normalized_score",
                        "env_for_normalization": args.env_name,
                        "d4rl_version": d4rl.__version__ if hasattr(d4rl, '__version__') else "unknown"
                    }
                    with open(os.path.join(best_save_dir, "best_eval_meta.json"), "w") as f:
                        json.dump(eval_meta, f, indent=2)

                    # Keep saving existing monitor_vals as before
                    if len(monitor_vals) > 0:
                        np.save(os.path.join(output_dir, f"{monitor_name}_best.npy"), monitor_vals)
                    np.save(os.path.join(output_dir, "best_eval"), evaluations)

            if len(offline_checkpoints) < actual_k:
                offline_checkpoints.append( (bc_loss_avg, curr_epoch) )
                offline_checkpoints.sort(key=lambda x: x[0])
                # Also save first best
                save_best_model_and_logs()
            else:
                # Already have actual_k items → replace worst
                worst_loss, worst_epoch = offline_checkpoints[-1]
                if bc_loss_avg < worst_loss:
                    offline_checkpoints[-1] = (bc_loss_avg, curr_epoch)
                    offline_checkpoints.sort(key=lambda x: x[0])
                    # Save if improved
                    save_best_model_and_logs()

        # 4.5 Early stopping check
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss_avg)
        metric = bc_loss_avg

    file_name_model = f"{args.env_name}_{args.exp}_{args.algo}"
    if args.save_model:
        model_save_dir = os.path.join("saved_models", file_name_model, f"seed{args.seed}")
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"Saving model to {model_save_dir} ...")
        agent.save_model(model_save_dir)

    if writer is not None:
        writer.close()
# ------------------------------------------------
# eval_policy: Evaluate on the standard environment
# ------------------------------------------------
def eval_policy(policy, env_name, seed, eval_episodes=10):
    """Evaluate a policy on the base (non-risky) environment.

    Args:
        policy: Policy object exposing sample_action(state) -> action.
        env_name (str): Gym environment name.
        seed (int): Random seed for evaluation env.
        eval_episodes (int): Number of evaluation episodes.

    Returns:
        tuple[float, float, float, float]: (avg_reward, std_reward, avg_norm_score, std_norm_score)
    """
    # Enable deterministic behavior only during evaluation
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # For generative policies (diffusion/flow), ensure deterministic internal randomness.
    # For flow-matching (e.g., RAFMAC), check deterministic mode in policy.sample_action()

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")

    # Restore non-deterministic behavior
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    return avg_reward, std_reward, avg_norm_score, std_norm_score

# ------------------------------------------------
# eval_policy_risky: Evaluate on risky-wrapped environment + count violations
# ------------------------------------------------
def eval_policy_risky(policy, env_name, seed, eval_episodes,risk_prob, risk_penalty, algo_name, dataset_name,max_vel,pro_vel_penal, cost_vel, prob_pose_penal, cost_pose, healthy_angle_range, done_if_exceed_factor):
    """Evaluate a policy on the risky-wrapped environment and log violations.

    Args:
        policy: Policy object exposing sample_action(state) -> action.
        env_name (str): Gym environment name.
        seed (int): Random seed for evaluation env.
        eval_episodes (int): Number of evaluation episodes.
        risk_prob (float): Probability of entering risky dynamics.
        risk_penalty (float): Penalty applied on risky events.
        algo_name (str): Algorithm label for logging.
        dataset_name (str): Dataset label for logging.
        max_vel (float): Velocity clamp for wrapper.
        pro_vel_penal (float): Probability of velocity penalty.
        cost_vel (float): Velocity penalty cost.
        prob_pose_penal (float): Probability of pose penalty.
        cost_pose (float): Pose penalty cost.
        healthy_angle_range (tuple[float, float]): Healthy angle range for wrapper.
        done_if_exceed_factor (float): Episode termination factor for exceeding limits.

    Returns:
        tuple: (avg_reward, std_reward, avg_norm_score, std_norm_score, violation_counts,
            episode_scores, monitor_vals, monitor_name, algo_name, dataset_name)
    """
    if not HAS_RISKY_WRAPPERS:
        raise RuntimeError("risky_wrappers not found. Install or place risky_wrappers.py.")

    # Enable deterministic behavior only during evaluation
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # For generative policies (diffusion/flow), ensure deterministic internal randomness.
    # For flow-matching (e.g., RAFMAC), check deterministic mode inside policy.sample_action()

    # Risky environment wrapper
    eval_env = make_risky_env(env_name, risk_prob=risk_prob, risk_penalty=risk_penalty, max_vel= max_vel,prob_vel_penal=pro_vel_penal, cost_vel=cost_vel, prob_pose_penal=prob_pose_penal, cost_pose=cost_pose, healthy_angle_range=healthy_angle_range, done_if_exceed_factor= done_if_exceed_factor)
    eval_env.seed(seed + 200)
    print(env.observation_space.shape, env.action_space.shape)
    scores = []
    violation_counts = 0
    total_steps = 0
     # Generic logging
    monitor_vals = []   # Accumulate velocity or angle, etc.
    monitor_name = None # velocity / angle / None

    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        ep_vel=[]
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            traj_return += reward
            if info.get('risky_state', False):
                violation_counts += 1
            total_steps += 1
             # Get log type and value
            if 'monitor_val' in info:
                monitor_vals.append(info['monitor_val'])
            if 'monitor_name' in info:
                # Should be same monitor_name across all steps; keep it
                monitor_name = info['monitor_name']
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    # Normalized score
    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)


    print_banner(
        f"[RISKY EVAL] Over {eval_episodes} episodes: Reward={avg_reward:.2f}, Norm={avg_norm_score:.2f}, Violation Counts={violation_counts:.2f}"
    )

    # Restore non-deterministic behavior
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    return avg_reward, std_reward, avg_norm_score, std_norm_score, violation_counts, scores, monitor_vals, monitor_name, algo_name, dataset_name


# ------------------------------------------------
# Main execution
# ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument("--env_name", default='walker2d-medium-expert-v2', type=str)
    parser.add_argument("--dir", default='results', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--config", default='configs', type=str,
                        help="Path to the configuration folder")

    ### Dataset Option ###
    parser.add_argument("--risky_dataset_path", default='', type=str,
                        help="Path to a risky dataset HDF5 or npy. If empty, use standard D4RL.")

    # Boolean flags with environment variable defaults
    parser.add_argument("--eval_risky_env", action='store_true',
                        default=False,
                        help="If set, evaluate the policy on a risky-wrapped env & measure violation counts.")
    parser.add_argument('--use_cvar', action='store_true',
                        default=False,
                        help='Enable IQN-based CVaR measurement in DiffusionQL')

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=int(os.getenv('BATCH_SIZE', '256')), type=int)
    parser.add_argument("--lr_decay", action='store_true',
                        default=False)
    parser.add_argument('--early_stop', action='store_true',
                        default=False)
    parser.add_argument('--save_best_model', action='store_true',
                        default=False)
    parser.add_argument('--record_tensorboard', action='store_true',
                        default=False)

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)

    ### Algo Choice ###
    parser.add_argument("--algo", default=os.getenv('ALGORITHM', 'radac'), type=str,
                        help="['bc', 'ql', 'radac', 'ddac']")

    parser.add_argument("--save_model", action='store_true',
                        default=False,
                        help="If set, save the trained model after finishing.")
    
    ### RAFMAC-specific options ###
    parser.add_argument("--use_distillation", action='store_true',
                        default=False,
                        help="If set, enable distillation training for RAFMAC (teacher-student learning).")

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    # Load hyperparameters from YAML configuration file for the selected environment
    hyperparameters = load_hyperparameters(env_name=args.env_name, config_path=os.path.join(args.config, f'{args.algo}.yaml'))

    args.num_epochs = 2000
    args.eval_freq = 50
    args.eval_episodes = 10 if 'v2' in args.env_name else 100

    args.lr_actor = float(hyperparameters['lr_actor'])
    args.lr_critic = float(hyperparameters['lr_critic'])
    args.eta = hyperparameters['eta']
    args.grad_norm = hyperparameters['grad_norm']
    args.top_k = hyperparameters['top_k']
    args.tau = hyperparameters.get('tau', 0.005)
    args.emb_dim = hyperparameters.get('emb_dim', 128)
    args.n_quantiles = hyperparameters.get('n_quantiles', 32)
    args.max_q_backup = hyperparameters.get('max_q_backup', False)
    args.q_clip_range = hyperparameters.get('q_clip_range', None)
    args.lambda_bc = hyperparameters.get('lambda_bc', 1.0)
    args.eta_warmup_steps = hyperparameters.get('eta_warmup_steps', 0)
    args.eta_ramp_steps = hyperparameters.get('eta_ramp_steps', 0)
    args.normalize_q_loss = hyperparameters.get('normalize_q_loss', False)
    args.q_agg = hyperparameters.get('q_agg', 'mean')

    # Hyperparameters for risky environment (use env vars as fallback if not in config)
    args.risk_distortion = hyperparameters.get('risk_distortion', 'cvar')
    args.max_vel = hyperparameters.get('max_vel', 2.0)
    args.prob_vel_penal = hyperparameters.get('prob_vel_penal', 0.0)
    args.cost_vel = hyperparameters.get('cost_vel', 0.0)
    args.risk_prob = hyperparameters.get('risk_prob', 0.0)
    args.risk_penalty = hyperparameters.get('risk_penalty', 0.0)
    args.prob_pose_penal = hyperparameters.get('prob_pose_penal', float(os.getenv('PROB_POSE_PENAL', '0.0')))
    args.cost_pose = hyperparameters.get('cost_pose', 0.0)
    args.healthy_angle_range = hyperparameters.get('healthy_angle_range',
                                                   (-0.5, 0.5))
    args.done_if_exceed_factor = hyperparameters.get('done_if_exceed_factor',
                                                     2.0)
    args.alpha = hyperparameters.get('alpha',0.1)
    args.start_steps = hyperparameters.get('start_steps', 1000)
    args.num_steps_per_epoch = 1000
    args.flow_steps = hyperparameters.get('flow_steps',10)
    
    # RAFMAC-specific parameters
    args.use_distillation = hyperparameters.get('use_distillation', False)
    
    # Compose logging-friendly file name
    file_name = f"{args.env_name}|{args.exp}|{args.algo}|T-{args.T}"
    if args.lr_decay:
        file_name += '|lr_decay'
    if args.save_best_model:
        file_name += f'|k-{args.top_k}'
    if args.risky_dataset_path:
        file_name += '|risky_data'
    if args.use_distillation:
        file_name += '|distill'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print_banner(f"Saving location: {results_dir}")

    # Set random seeds
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)

    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    # Start training
    train_agent(env, state_dim, action_dim, max_action, args.device, results_dir, args)
