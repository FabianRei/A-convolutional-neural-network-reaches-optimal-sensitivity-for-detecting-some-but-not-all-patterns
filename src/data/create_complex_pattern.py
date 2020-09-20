import numpy as np
import os


# creates a cellular automaton pattern (https://en.wikipedia.org/wiki/Cellular_automaton)
def get_applied_rule(row, rule_dict):
    lcr = np.vstack((np.roll(row, 1), row, np.roll(row, -1)))
    new_row = np.apply_along_axis(lambda x: rule_dict[''.join(x.astype(str))], axis=0, arr=lcr)
    return new_row


def create_automaton(rule=110, size=(238, 238), seed=42):
    np.random.seed(seed)
    bin_rule_list = [int(b) for b in np.binary_repr(rule, 8)]
    pattern_list = [np.binary_repr(pat, 3) for pat in range(7, -1, -1)]
    rule_dict = {key: val for key, val in zip(pattern_list, bin_rule_list)}
    result = np.zeros(size, dtype=np.int)
    result[0, :] = np.random.randint(2, size=size[1])
    for row in range(result.shape[0]-1):
        result[row + 1] = get_applied_rule(result[row], rule_dict)
    return result


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import scipy.misc
    out_path = r'C:\Users\Fabian\Documents\data\automatons_cutouts'
    os.makedirs(out_path, exist_ok=True)
    shuffle_cols = False
    cutout = 60
    for i in range(256):
        automaton = create_automaton(rule=i)
        if shuffle_cols:
            np.random.seed(41)
            shuff_idxs = np.random.permutation(automaton.shape[1])
            automaton = np.take(automaton, shuff_idxs, axis=1)
        if cutout > -1:
            automaton = automaton[:cutout, :cutout]
        if shuffle_cols:
            scipy.misc.imsave(f"{out_path}\\automaton_rule_{i}_shuff_cols.png", automaton)
        else:
            scipy.misc.imsave(f"{out_path}\\automaton_rule_{i}.png", automaton)
    # image = create_automaton()
    # plt.imshow(image, cmap='gray')
    # print('done')
