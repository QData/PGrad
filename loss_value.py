import numpy as np
import json
import glob
import os
import matplotlib.pyplot as plt
import argparse



def load_losses(data_path, algorithm):

    ALL_files = glob.glob(data_path+ '/*')
    ALL_LOSSES = []
    for files in ALL_files:
        
        if os.path.exists(files+'/done'):
            with open(files+'/results.jsonl') as f:
                data = [json.loads(line) for line in f]
            if data[0]['args']['algorithm'] == algorithm:
                
                ALL_LOSSES.append([])
                for step in range(len(data)):
                    ALL_LOSSES[-1].append([data[step]['env0_in_acc'], data[step]['env1_in_acc'], data[step]['env2_in_acc'], data[step]['env3_in_acc'], data[step]['args']['test_envs'][0], data[step]['step']])

    all_losses = np.array(ALL_LOSSES)
    return all_losses




##### start slicing ##########
def slicing_losses(all_losses, test_env):
    index_0 = np.unique(np.where(all_losses[:,:,4]==test_env)[0])
    domain_losses = all_losses[index_0]
    mean_losses = np.mean(domain_losses, axis=0)
    var_losses = np.std(domain_losses, axis=0)
    return mean_losses, var_losses


def visualization(mean_losses, var_losses, num_domains, test_env, algorithm):
    domains = ['V', 'L', 'C', "S"]
    colors = ['g', 'r', 'b', 'c']
    plt.figure()
    index = np.linspace(0, 5000, num=51)[10:]
    for i in range(num_domains):
        if i != test_env:
            plt.grid()
            plt.plot(index, mean_losses[10:, i], linewidth=2, color=colors[i])
            plt.fill_between(index, mean_losses[10:, i]+var_losses[10:, i], mean_losses[10:, i]-var_losses[10:, i], alpha=0.3, color=colors[i])
    domains.pop(test_env)
    plt.legend(domains)
    plt.tight_layout()
    plt.savefig('./Loss_curves/'+algorithm+'/'+str(test_env)+'.png')




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--data_dir', type=str, default='./visual_VLCS')
    parser.add_argument('--test_env', type=int, default=0)
    args = parser.parse_args()

    args.num_domains = 4

    all_losses = load_losses(args.data_dir, args.algorithm)
    mean_losses, var_losses = slicing_losses(all_losses, args.test_env)
    visualization(mean_losses, var_losses, args.num_domains, args.test_env, args.algorithm)

