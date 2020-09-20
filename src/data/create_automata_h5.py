from deepLearning.src.data.create_complex_pattern import create_automaton
import h5py
import numpy as np
import os



def create_automata_h5(out_path, rule=110, a_class='noClass', size=(512, 512), for_matlab=True):
    size = (256, 256)
    automaton = create_automaton(rule=rule, size=size, seed=42)
    if for_matlab:
        automaton = automaton / (automaton.std()/0.7071)
        automaton -= automaton.mean()
        fname= f"automata_rule_{rule}_{a_class}.h5"
        os.makedirs(out_path, exist_ok=True)
        save_path = os.path.join(out_path, fname)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset(name='face_mat', data=automaton)
    else:
        folder_name = f"plain_automata_rule_{rule}_{a_class}"
        out_path = os.path.join(out_path, folder_name)
        os.makedirs(out_path, exist_ok=True)
        automaton = automaton / (automaton.std()/0.7071)
        automaton -= automaton.mean()
        mean_photons = 331.0224
        automaton *= 203.1721
        contrasts = np.logspace(-5, -1.7, 12)
        no_signal_mean = np.ones(automaton.shape)* mean_photons
        for c in contrasts:
            noNoise = []
            noNoise.append(no_signal_mean)
            noNoise.append(automaton*c + mean_photons)
            noNoise = np.array(noNoise)
            noNoise = np.transpose(noNoise, (0, 2, 1))
            noise = np.random.poisson(noNoise)
            fname = f"automata_rule_{rule}_{a_class}_contrast_{c:.8f}.h5"
            save_path = os.path.join(out_path, fname)
            with h5py.File(save_path, 'w') as f:
                f.create_dataset(name='noNoiseImg', data=noNoise)
                f.create_dataset(name='imgNoise', data=noise)
                f.create_dataset(name='noNoiseImgContrast', data=[0, c])
                f.create_dataset(name='imgNoiseContrasts', data=[0, c])
                f.create_dataset(name='noNoiseImgFreq', data=1)




if __name__ == '__main__':
    p = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\redo_automaton\plain_automata'
    class_2_rules = [3, 57, 76, 78]
    class_3_rules = [22, 30, 75, 101]
    no_class_rules = [45, 105, 110, 154]
    for rule in class_2_rules:
        create_automata_h5(p, rule, a_class='class2', for_matlab=False)
    for rule in class_3_rules:
        create_automata_h5(p, rule, a_class='class3', for_matlab=False)
    for rule in no_class_rules:
        create_automata_h5(p, rule, for_matlab=False)


r'''
Past runs:
################################################################
if __name__ == '__main__':
    p = r'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\redo_automaton\matlab_templates'
    class_2_rules = [3, 57, 76, 78]
    class_3_rules = [22, 30, 75, 101]
    no_class_rules = [45, 105, 110, 154]
    for rule in class_2_rules:
        create_automata_h5(p, rule, a_class='class2')
    for rule in class_3_rules:
        create_automata_h5(p, rule, a_class='class3')
    for rule in no_class_rules:
        create_automata_h5(p, rule)
'''