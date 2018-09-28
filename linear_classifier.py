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
weight_matrix = [];
valtoname = {};

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

def Traindata(iterations,learn_rate,reg):
	with open(train_data_path) as file:
		text = file.read().splitlines();
	train_data_len = len(text);
	train_arr = [];
	size = 32;
	mean_train = [0]*size*size;
	mean_img = [[0]*32]*32;
	value_intial = 0;
	y = [];
	for entry in text:
		class_image = entry.split(' ')[1];
		file_name = entry.split(' ')[0];
		if class_image not in img_dict:
			img_dict[class_image] = {};
			img_dict[class_image]['images'] = [];
			img_dict[class_image]['value'] = [];
			img_dict[class_image]['value'] = value_intial;
			valtoname[value_intial] = class_image;
			value_intial += 1;
		y.append(img_dict[class_image]['value']);
		temp = load_image(file_name);#to send it into corresponding class
		img_dict[class_image]['images'].append(temp);
		mean_train += (temp/train_data_len);
		train_arr.append(temp);
	print(y)
	print(valtoname)
	for i in img_dict:
		img_dict[i]['value'] = [];
		img_dict[i]['value'] = value_intial;
		value_intial += 1;
		# print(img_dict[i])
		# print(img_dict[i]['value'])# this is to assign numeric values for each class
	no_of_classes = value_intial;
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
	reduced_img = np.matmul(train_np,k_components.transpose());
	reduced_img = reduced_img.transpose();
	# print(reduced_img.shape);
	dim = reduced_img.shape[0];
	weight_matrix =  np.random.randn(no_of_classes, dim) * 0.001
	print(weight_matrix.shape);
	losses_history = [];
	for i in range(iterations):
		loss = 0;
		grad = np.zeros_like(weight_matrix);
		dim, num_train = reduced_img.shape;
		scores = np.matmul(weight_matrix,reduced_img) # [K, N]
		# Shift scores so that the highest value is 0
		scores -= np.max(scores)
		scores_exp = np.exp(scores)
		correct_scores_exp = scores_exp[y, range(num_train)] # [N, ]
		scores_exp_sum = np.sum(scores_exp, axis=0) # [N, ]
		loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
		loss /= num_train
		loss += 0.5 * reg * np.sum(weight_matrix*weight_matrix)
		scores_exp_normalized = scores_exp / scores_exp_sum
		# deal with the correct class
		scores_exp_normalized[y, range(num_train)] -= 1 # [K, N]
		grad = scores_exp_normalized.dot(reduced_img.transpose())
		grad /= num_train
		grad += reg * weight_matrix
		losses_history.append(loss);
		weight_matrix -= learn_rate * grad # [K x D]
	# print(weight_matrix);
	return img_dict,k_components,losses_history,reduced_img,weight_matrix;

def TestData(train_dict,k_eig_components,weight_matrix):
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
		scores = weight_matrix.dot(final_test)
		pred_ys = np.argmax(scores, axis=0)
		pred_y = valtoname[pred_ys];
		answer.append(pred_y)
	return answer;

train_dict,k_eig_components,losses_history,reduced_img,weight_matrix = Traindata(1000,0.000001,100);
print(losses_history)
answer = TestData(train_dict,k_eig_components,weight_matrix);
print(answer)
with open('output_file.txt', 'w') as f:
    for item in answer:
        f.write("%s" % item)



