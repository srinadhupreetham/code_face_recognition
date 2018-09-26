import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt;


tot_train_images = len(glob.glob('./dataset/*.jpg'));
train_arr = [];
mean_train = [0]*1024;
mean_img = [[0]*32]*32;
train_list = glob.glob('./dataset/*.jpg');
def load_image(x):
	img = Image.open(x).convert('L');
	img = img.resize((32,32), Image.BILINEAR);
	imgarr = np.array(img);
	flat = imgarr.flatten();
	return flat

for infile in train_list:
	temp = load_image(infile);
	mean_train += (temp/tot_train_images);
	train_arr.append(temp);

train_np = np.array(train_arr);
mean_arr = [mean_train] * tot_train_images;
cov_matrix = np.subtract(train_np,mean_arr);
L_matrix = np.matmul(cov_matrix.transpose(),cov_matrix);
eig_val,eig_vec = np.linalg.eig(L_matrix);
k_components_val = 64;

eigen_pairs = []
for egn in range(len(eig_val)):
	eigen_pairs.append((eig_val[egn], eig_vec[:,egn]))

eigen_pairs.sort(reverse=True);
# eigen_pairs = np.array(eigen_pairs);
eig_val.sort();
img1 = train_arr[1];

sorted_eigenvectors = np.zeros((32*32, 32*32))
sorted_eigenvalues = np.zeros((32*32, 1))

for egn in range(len(eig_val)):
    sorted_eigenvalues[egn] = eigen_pairs[egn][0]
    sorted_eigenvectors[:,egn] = eigen_pairs[egn][1]

k_components = sorted_eigenvectors.transpose()
k_components = k_components[:1024]

alpha_vec = np.matmul(k_components,img1);
# image reconstruction start
reconstructed_image = np.zeros((32*32,1));
for alpha in range(len(alpha_vec)):
	temp = alpha_vec[alpha]*k_components[alpha];
	temp = temp.reshape(temp.size,1);
	reconstructed_image = reconstructed_image + (temp) ;

final = reconstructed_image.reshape((32,32));
# image reconstruction ends
print(final);
final_temp = np.array(final);
final_temp = Image.fromarray(final_temp);
final_temp = final_temp.resize((256,256),Image.BILINEAR);
plt.imshow(final_temp, cmap='gray')
plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()
