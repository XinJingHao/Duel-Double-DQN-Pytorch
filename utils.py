def evaluate_policy(env, model, render, turns = 3):
    scores = 0
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, deterministic=True)
            s_prime, r, done, info = env.step(a)
            ep_r += r
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return int(scores/turns)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')