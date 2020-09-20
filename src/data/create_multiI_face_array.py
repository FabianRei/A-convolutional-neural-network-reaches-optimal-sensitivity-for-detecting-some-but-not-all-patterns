import h5py
import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2


h5_face_path = r'C:\Users\Fabian\Documents\data\faces\multi_face_experiment'
h5_faces = glob(h5_face_path + r'\*.h5')
print(h5_faces)
h5_arrs = []
for face in h5_faces:
    with h5py.File(face, 'r') as f:
        h5_arrs.append(np.array(f['face_mat']))
result = np.zeros((1536, 1536))
w = 512
for i in range(3):
    for j in range(3):
        result[i*w:(i+1)*w, j*w:(j+1)*w] = h5_arrs[i*3+j]

result = cv2.resize(result, (512, 512))
save_dir = os.path.join(h5_face_path, 'multi_face_result.h5')
with h5py.File(save_dir, 'w') as f:
    f.create_dataset(name='face_mat', data=result)
# plt.imshow(result)

