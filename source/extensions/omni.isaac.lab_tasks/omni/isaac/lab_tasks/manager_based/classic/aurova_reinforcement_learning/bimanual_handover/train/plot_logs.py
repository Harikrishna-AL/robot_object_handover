from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

def get_rewards(event_files):
    all_rewards = []
    all_timesteps = []
    
    for files in event_files:
        rewards = []
        timesteps = []

        for file in files:
            ea = event_accumulator.EventAccumulator(file)
            ea.Reload()
            scalars = ea.Scalars('rollout/ep_rew_mean')  

            initial_time = scalars[0].step
            timesteps_run = [(scalar.step - initial_time) for scalar in scalars[4:]]  # Skip first 4 entries
            rewards_run = [scalar.value for scalar in scalars[4:]]

            if len(timesteps_run) > 0:
                timesteps.append(timesteps_run)
                rewards.append(rewards_run)

        # Align data (truncate to shortest length)
        min_length = min(map(len, timesteps)) if timesteps else 0
        timesteps = np.array([t[:min_length] for t in timesteps])
        rewards = np.array([r[:min_length] for r in rewards])

        all_timesteps.append(np.mean(timesteps, axis=0) if timesteps.size else [])
        all_rewards.append(rewards)

    return all_rewards, all_timesteps

def plot_results(event_files):
    rewards, timesteps = get_rewards(event_files.values())

    plt.figure(figsize=(6, 4))
    colors = ['blue', 'red', 'green']
    labels = list(event_files.keys())

    for i, (reward, timestep) in enumerate(zip(rewards, timesteps)):
        if len(reward) > 0:
            mean_reward = np.mean(reward, axis=0)
            std_reward = np.std(reward, axis=0)

            plt.plot(timestep, mean_reward, label=labels[i], color=colors[i])
            plt.fill_between(timestep, mean_reward - std_reward, mean_reward + std_reward, color=colors[i], alpha=0.3)

    plt.xlabel("Time Step", fontsize=28)
    plt.ylabel("Average Reward", fontsize=28)
    

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ticklabel_format(
                    axis="x",
                    style="sci",
                    scilimits=(0,0),
                    useMathText=False
)
    plt.gca().xaxis.get_offset_text().set_fontsize(20)  # or whatever size you want
    plt.rc('font', size=20)
    plt.legend(fontsize=20)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    event_files = {
        "DQ": [
            "logs/2025-02-14_08-38-27/events.out.tfevents.1739522406.asimovgpu.3600294.0",
            "logs/2025-02-19_15-46-39/events.out.tfevents.1739980119.asimovgpu.32.0"
        ],
        "EULER": [
            "logs/2025-02-24_13-22-51/events.out.tfevents.1740403466.asimovgpu.652182.0",
            "logs/2025-02-24_13-26-39/events.out.tfevents.1740403703.asimovgpu.796827.0"
        ], 
        "MATRIX": 
        [
            "logs/2025-02-28_11-23-51/events.out.tfevents.1740741924.asimovgpu.2502138.0",
            "logs/2025-02-28_11-23-51/events.out.tfevents.1740741924.asimovgpu.2502138.0",
        ]
    }

    plot_results(event_files)
