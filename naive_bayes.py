import numpy as np
from PIL import Image
import glob
import sys
import matplotlib.pyplot as plt;
import math

train_data_path = str(sys.argv[1]);
test_data_path = str(sys.argv[2]);
img_dict = {};
train_arr = [];

def getNormalDistValue(val, mean, variance):
    ans = 1/math.sqrt(2*math.pi*variance);
    exp_val = -((val-mean)*(val-mean))/(2*variance);
    exp_val = math.exp(exp_val);
    ans *= exp_val;
    return ans
def load_image(x):
	img = Image.open(x).convert('L');
	img = img.resize((32,32), Image.BILINEAR);
	imgarr = np.array(img);
	flat = imgarr.flatten();
	return flat
def Traindata():
	with open(train_data_path) as file:
		text = file.readlines();
	train_data_len = len(text);
	train_arr = [];
	size = 32;
	mean_train = [0]*size*size;
	mean_img = [[0]*32]*32;
	for entry in text:
		class_image = entry.split(' ')[1];
		file_name = entry.split(' ')[0];
		if class_image not in img_dict:
			img_dict[class_image] = {};
			img_dict[class_image]['images'] = [];
		temp = load_image(file_name);#to send it into corresponding class
		img_dict[class_image]['images'].append(temp);
		mean_train += (temp/train_data_len);
		train_arr.append(temp);
	train_np = np.array(train_arr);
	mean_arr = [mean_train] * train_data_len;
	cov_matrix = np.subtract(train_np,mean_arr);
	L_matrix = np.matmul(cov_matrix.transpose(),cov_matrix);
	eig_val,eig_vec = np.linalg.eig(L_matrix);
	k_components_val = 64;
	eigen_pairs = [];
	for egn in range(len(eig_val)):
		eigen_pairs.append((eig_val[egn], eig_vec[:,egn]));
	eigen_pairs.sort(reverse=True);
	eig_val.sort();
	img1 = train_arr[1];
	sorted_eigenvectors = np.zeros((32*32, 32*32));
	sorted_eigenvalues = np.zeros((32*32, 1));
	for egn in range(len(eig_val)):
		sorted_eigenvalues[egn] = eigen_pairs[egn][0];
		sorted_eigenvectors[:,egn] = eigen_pairs[egn][1];
	k_components = sorted_eigenvectors.transpose();
	k_components = k_components[:k_components_val];
	for im_cls in img_dict:
		images = img_dict[im_cls]['images'];
		each_cls_images = len(images);
		img_dict[im_cls]['prob'] = each_cls_images/train_data_len;
		mean_class =  [0]*k_components_val;
		final_class = np.matmul(images,k_components.transpose());
		for temp_img in final_class:
			mean_class = mean_class + (temp_img/each_cls_images);
		img_dict[im_cls]['mean'] = mean_class;
		var_arr = np.array(final_class);
		img_dict[im_cls]['variance'] = np.var(final_class,axis = 0);
	return img_dict,k_components;

def TestData(train_dict,k_eig_components):
	with open(test_data_path) as file:
		text = file.read().splitlines();
	test_data_len = len(text);
	size = 32;
	answer = [];
	for entry in text:
		file_name = entry;
		img = Image.open(file_name).convert('L');
		img = img.resize((size,size), Image.BILINEAR);
		imgarr = np.array(img);
		prob = 0;
		flat = imgarr.flatten().reshape(1,size*size);
		final_testi = np.matmul(flat,k_eig_components.transpose());
		final_test = final_testi[0];
		final_test_len = len(final_test);
		for im_cls in train_dict:
			mean = train_dict[im_cls]['mean'];
			variance = train_dict[im_cls]['variance'];
			new_prob = 0;temp_product = 1;
			for i in range(final_test_len):
				temp_product = temp_product * getNormalDistValue(final_test[i], mean[i], variance[i]);
			new_prob = temp_product;
			if(new_prob >= prob):
				prob = new_prob;
				final_answer = im_cls;
		answer.append(final_answer);
		print(final_answer);
	return answer;

train_dict,k_eig_components = Traindata();
answer = TestData(train_dict,k_eig_components);


