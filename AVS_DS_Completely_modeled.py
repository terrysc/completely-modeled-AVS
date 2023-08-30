#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numba
from PIL import Image
import pdb

np.set_printoptions(linewidth = 500)  
np.set_printoptions(threshold = np.inf) 


threshold = 0

@numba.jit

def AVS_DS_Color_1(image_1, image_2): # image_1 is image inputed
	h_image, w_image, colorChannel_num = image_1.shape  
	GMDNs_3Channel_out = np.zeros((8, colorChannel_num)) # GMDNs_1Channel_out is the detection result (1,0,0,0,0,0,0,0)
	GMDNs_out = np.zeros(8)
	LMDNs_3Channel_out = np.zeros((8, h_image, w_image, colorChannel_num))
	for colorChannel in range(colorChannel_num):
		for i in range(1, h_image-1):
			for j in range(1, w_image-1):
				LRF_image1 = image_1[i-1:i+2, j-1:j+2, colorChannel]
				LRF_image2 = image_2[i-1:i+2, j-1:j+2, colorChannel]
				LRF_C_image1 = LRF_image1[1, 1]
				LRF_C_image2 = LRF_image2[1, 1]

				LRF_R_image1, LRF_R_image2 = LRF_image1[1, 2], LRF_image2[1, 2]
				LRF_UR_image1, LRF_UR_image2 = LRF_image1[0, 2], LRF_image2[0, 2]
				LRF_U_image1, LRF_U_image2 = LRF_image1[0, 1], LRF_image2[0, 1]
				LRF_UL_image1, LRF_UL_image2 = LRF_image1[0, 0], LRF_image2[0, 0]
				LRF_L_image1, LRF_L_image2 = LRF_image1[1, 0], LRF_image2[1, 0]
				LRF_LL_image1, LRF_LL_image2 = LRF_image1[2, 0], LRF_image2[2, 0]
				LRF_D_image1, LRF_D_image2 = LRF_image1[2, 1], LRF_image2[2, 1]
				LRF_LR_image1, LRF_LR_image2 = LRF_image1[2, 2], LRF_image2[2, 2]

				if abs(LRF_C_image2 - LRF_C_image1) > threshold:
					if abs(LRF_R_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[0, i, j, colorChannel] = 1 # rightwards
					if abs(LRF_UR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[1, i, j, colorChannel] = 1 # upper rightwards
					if abs(LRF_U_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[2, i, j, colorChannel] = 1 # upwards
					if abs(LRF_UL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[3, i, j, colorChannel] = 1 # upper leftwards
					if abs(LRF_L_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[4, i, j, colorChannel] = 1 # leftwards
					if abs(LRF_LL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[5, i, j, colorChannel] = 1 # lower leftwards
					if abs(LRF_D_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[6, i, j, colorChannel] = 1 # downwards
					if abs(LRF_LR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[7, i, j, colorChannel] = 1 # lower rightward
				# else:
				# 	if abs(LRF_R_image2 - LRF_R_image1) > threshold:
				# 		if abs(LRF_R_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[0, i, j, colorChannel] = 1 # rightwards
				# 	if abs(LRF_UR_image2 - LRF_UR_image1) > threshold:
				# 		if abs(LRF_UR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[1, i, j, colorChannel] = 1 # upper rightwards
				# 	if abs(LRF_U_image2 - LRF_U_image1) > threshold:
				# 		if abs(LRF_U_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[2, i, j, colorChannel] = 1 # upwards
				# 	if abs(LRF_UL_image2 - LRF_UL_image1) > threshold:
				# 		if abs(LRF_UL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[3, i, j, colorChannel] = 1 # upper leftwards
				# 	if abs(LRF_L_image2 - LRF_L_image1) > threshold:
				# 		if abs(LRF_L_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[4, i, j, colorChannel] = 1 # leftwards
				# 	if abs(LRF_LL_image2 - LRF_LL_image1) > threshold:
				# 		if abs(LRF_LL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[5, i, j, colorChannel] = 1 # lower leftwards
				# 	if abs(LRF_D_image2 - LRF_D_image1) > threshold:
				# 		if abs(LRF_D_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[6, i, j, colorChannel] = 1 # downwards
				# 	if abs(LRF_LR_image2 - LRF_LR_image1) > threshold:
				# 		if abs(LRF_LR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[7, i, j, colorChannel] = 1 # lower rightward
		
		for i in range(8):
			GMDNs_3Channel_out[i, colorChannel] = np.sum(LMDNs_3Channel_out[i, :, :, colorChannel])
	for i in range(8):
		GMDNs_out[i] = np.sum(GMDNs_3Channel_out[i, :])

	return GMDNs_out # (8,)

def AVS_DS_Color_2(image_1, image_2): # image_1 is image inputed
	h_image, w_image, colorChannel_num = image_1.shape  
	GMDNs_3Channel_out = np.zeros((8, colorChannel_num)) # GMDNs_1Channel_out is the detection result (1,0,0,0,0,0,0,0)
	GMDNs_out = np.zeros(8)
	LMDNs_3Channel_out = np.zeros((8, h_image, w_image, colorChannel_num))
	for colorChannel in range(colorChannel_num):
		for i in range(1, h_image-1):
			for j in range(1, w_image-1):
				LRF_image1 = image_1[i-1:i+2, j-1:j+2, colorChannel]
				LRF_image2 = image_2[i-1:i+2, j-1:j+2, colorChannel]
				LRF_C_image1 = LRF_image1[1, 1]
				LRF_C_image2 = LRF_image2[1, 1]

				LRF_R_image1, LRF_R_image2 = LRF_image1[1, 2], LRF_image2[1, 2]
				LRF_UR_image1, LRF_UR_image2 = LRF_image1[0, 2], LRF_image2[0, 2]
				LRF_U_image1, LRF_U_image2 = LRF_image1[0, 1], LRF_image2[0, 1]
				LRF_UL_image1, LRF_UL_image2 = LRF_image1[0, 0], LRF_image2[0, 0]
				LRF_L_image1, LRF_L_image2 = LRF_image1[1, 0], LRF_image2[1, 0]
				LRF_LL_image1, LRF_LL_image2 = LRF_image1[2, 0], LRF_image2[2, 0]
				LRF_D_image1, LRF_D_image2 = LRF_image1[2, 1], LRF_image2[2, 1]
				LRF_LR_image1, LRF_LR_image2 = LRF_image1[2, 2], LRF_image2[2, 2]

				if abs(LRF_C_image2 - LRF_C_image1) > threshold:
					if abs(LRF_R_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[0, i, j, colorChannel] = 1 # rightwards
					if abs(LRF_UR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[1, i, j, colorChannel] = 1 # upper rightwards
					if abs(LRF_U_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[2, i, j, colorChannel] = 1 # upwards
					if abs(LRF_UL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[3, i, j, colorChannel] = 1 # upper leftwards
					if abs(LRF_L_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[4, i, j, colorChannel] = 1 # leftwards
					if abs(LRF_LL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[5, i, j, colorChannel] = 1 # lower leftwards
					if abs(LRF_D_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[6, i, j, colorChannel] = 1 # downwards
					if abs(LRF_LR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[7, i, j, colorChannel] = 1 # lower rightward
				else:
					if abs(LRF_R_image2 - LRF_R_image1) > threshold:
						if abs(LRF_R_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[0, i, j, colorChannel] = 1 # rightwards
					if abs(LRF_UR_image2 - LRF_UR_image1) > threshold:
						if abs(LRF_UR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[1, i, j, colorChannel] = 1 # upper rightwards
					if abs(LRF_U_image2 - LRF_U_image1) > threshold:
						if abs(LRF_U_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[2, i, j, colorChannel] = 1 # upwards
					if abs(LRF_UL_image2 - LRF_UL_image1) > threshold:
						if abs(LRF_UL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[3, i, j, colorChannel] = 1 # upper leftwards
					if abs(LRF_L_image2 - LRF_L_image1) > threshold:
						if abs(LRF_L_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[4, i, j, colorChannel] = 1 # leftwards
					if abs(LRF_LL_image2 - LRF_LL_image1) > threshold:
						if abs(LRF_LL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[5, i, j, colorChannel] = 1 # lower leftwards
					if abs(LRF_D_image2 - LRF_D_image1) > threshold:
						if abs(LRF_D_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[6, i, j, colorChannel] = 1 # downwards
					if abs(LRF_LR_image2 - LRF_LR_image1) > threshold:
						if abs(LRF_LR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[7, i, j, colorChannel] = 1 # lower rightward
		
		for i in range(8):
			GMDNs_3Channel_out[i, colorChannel] = np.sum(LMDNs_3Channel_out[i, :, :, colorChannel])
	for i in range(8):
		GMDNs_out[i] = np.sum(GMDNs_3Channel_out[i, :])

	return GMDNs_out # (8,)

def AVS_DS_Color_2(image_1, image_2): # image_1 is image inputed
	h_image, w_image, colorChannel_num = image_1.shape  
	GMDNs_3Channel_out = np.zeros((8, colorChannel_num)) # GMDNs_1Channel_out is the detection result (1,0,0,0,0,0,0,0)
	GMDNs_out = np.zeros(8)
	LMDNs_3Channel_out = np.zeros((8, h_image, w_image, colorChannel_num))
	for colorChannel in range(colorChannel_num):
		for i in range(1, h_image-1):
			for j in range(1, w_image-1):
				LRF_image1 = image_1[i-1:i+2, j-1:j+2, colorChannel]
				LRF_image2 = image_2[i-1:i+2, j-1:j+2, colorChannel]
				LRF_C_image1 = LRF_image1[1, 1]
				LRF_C_image2 = LRF_image2[1, 1]

				LRF_R_image1, LRF_R_image2 = LRF_image1[1, 2], LRF_image2[1, 2]
				LRF_UR_image1, LRF_UR_image2 = LRF_image1[0, 2], LRF_image2[0, 2]
				LRF_U_image1, LRF_U_image2 = LRF_image1[0, 1], LRF_image2[0, 1]
				LRF_UL_image1, LRF_UL_image2 = LRF_image1[0, 0], LRF_image2[0, 0]
				LRF_L_image1, LRF_L_image2 = LRF_image1[1, 0], LRF_image2[1, 0]
				LRF_LL_image1, LRF_LL_image2 = LRF_image1[2, 0], LRF_image2[2, 0]
				LRF_D_image1, LRF_D_image2 = LRF_image1[2, 1], LRF_image2[2, 1]
				LRF_LR_image1, LRF_LR_image2 = LRF_image1[2, 2], LRF_image2[2, 2]

				if abs(LRF_C_image2 - LRF_C_image1) > threshold:
					if abs(LRF_R_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[0, i, j, colorChannel] = 1 # rightwards
					if abs(LRF_UR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[1, i, j, colorChannel] = 1 # upper rightwards
					if abs(LRF_U_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[2, i, j, colorChannel] = 1 # upwards
					if abs(LRF_UL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[3, i, j, colorChannel] = 1 # upper leftwards
					if abs(LRF_L_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[4, i, j, colorChannel] = 1 # leftwards
					if abs(LRF_LL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[5, i, j, colorChannel] = 1 # lower leftwards
					if abs(LRF_D_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[6, i, j, colorChannel] = 1 # downwards
					if abs(LRF_LR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[7, i, j, colorChannel] = 1 # lower rightward
				# else:
				# 	if abs(LRF_R_image2 - LRF_R_image1) > threshold:
				# 		if abs(LRF_R_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[0, i, j, colorChannel] = 1 # rightwards
				# 	if abs(LRF_UR_image2 - LRF_UR_image1) > threshold:
				# 		if abs(LRF_UR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[1, i, j, colorChannel] = 1 # upper rightwards
				# 	if abs(LRF_U_image2 - LRF_U_image1) > threshold:
				# 		if abs(LRF_U_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[2, i, j, colorChannel] = 1 # upwards
				# 	if abs(LRF_UL_image2 - LRF_UL_image1) > threshold:
				# 		if abs(LRF_UL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[3, i, j, colorChannel] = 1 # upper leftwards
				# 	if abs(LRF_L_image2 - LRF_L_image1) > threshold:
				# 		if abs(LRF_L_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[4, i, j, colorChannel] = 1 # leftwards
				# 	if abs(LRF_LL_image2 - LRF_LL_image1) > threshold:
				# 		if abs(LRF_LL_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[5, i, j, colorChannel] = 1 # lower leftwards
				# 	if abs(LRF_D_image2 - LRF_D_image1) > threshold:
				# 		if abs(LRF_D_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[6, i, j, colorChannel] = 1 # downwards
				# 	if abs(LRF_LR_image2 - LRF_LR_image1) > threshold:
				# 		if abs(LRF_LR_image2 - LRF_C_image1) <= threshold: LMDNs_3Channel_out[7, i, j, colorChannel] = 1 # lower rightward
		
		for i in range(8):
			GMDNs_3Channel_out[i, colorChannel] = np.sum(LMDNs_3Channel_out[i, :, :, colorChannel])
	for i in range(8):
		GMDNs_out[i] = np.sum(GMDNs_3Channel_out[i, :])

	return GMDNs_out # (8,)
	




## main.py ##

Alg_name = 'AVS_DS_Color_HC_AC_' + str(threshold)
data_fileName = 'DS_Color_2Images_Data'
# Alg_name = 'AVS_DS_Color_HC_' + str(threshold)
# data_name_list = ['DS_Color_2Images_Data_noisetype0_', 'DS_Color_2Images_Data_noisetype1_', 'DS_Color_2Images_Data_noisetype2_', 'DS_Color_2Images_Data_noisetype3_', 'DS_Color_2Images_Data_noisetype4_']
# data_name_list = ['DS_Color_3Images_Data_noisetype0_', 'DS_Color_3Images_Data_noisetype1_', 'DS_Color_3Images_Data_noisetype2_', 'DS_Color_3Images_Data_noisetype3_', 'DS_Color_3Images_Data_noisetype4_']
# data_name_list = ['DS_Binary_2Images_Data_noisetype4_']
data_name_list = ['DS_Color_2Images_Data_noisetype0_', 'DS_Color_2Images_Data_noisetype1_', 'DS_Color_2Images_Data_noisetype2_', 'DS_Color_2Images_Data_noisetype3_', 'DS_Color_2Images_Data_noisetype4_']
for data_name in data_name_list:
	# datascale = 2000
	datascale = 200
	imagescale = 1024
	#noise_proportion_list = [0, 0.1, 0.2, 0.3]
	if data_name == 'DS_Color_2Images_Data_noisetype0_':
		noise_proportion_list = [0]
	else:
		noise_proportion_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		noise_proportion_list = [0.1, 0.2, 0.3]
		# noise_proportion_list = [0.3]

	for noise_proportion in noise_proportion_list:
		writer = pd.ExcelWriter('./' + Alg_name + '_' + data_name + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '.xlsx')
		object_scale_list = [1,2,4,8,16,32,64,128,256,512]
		#object_scale_list = [32,2,4,8,16,32]
		Num_dataset = 0
		result_excel = np.zeros([len(object_scale_list), 3])
		index = []
		for object_scale in object_scale_list:
			#writer = pd.ExcelWriter('./' + Alg_name + '_' + data_name + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '.xlsx')
			#data = np.load('/Volumes/Motion data/dataset/2D motion/32x32 size/p noise 2/x_128pixel_0%contactpnoise.npy')
			#label = np.load('/Volumes/Motion data/dataset/2D motion/32x32 size/p noise 2/t_128pixel_0%contactpnoise.npy')
			index = index + [str(object_scale)]

			Dir_path = os.getcwd()
			data = np.load(str(Dir_path) + '/' + data_fileName + '/' + data_name + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_x' + '.npy')
			label = np.load(str(Dir_path) + '/' + data_fileName + '/' + data_name + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_t' + '.npy')
			print(data.shape)
			sample_num, frame_num, h_image, w_image, colorChannel_num = data.shape # (10000, 2, 32, 32)/RGB(10000, 2, 32, 32, 3)
			result = np.zeros((sample_num, 8))
			for i in range(sample_num):
				image_1 = data[i,0]  #  image_1 = data[i,0,:,:]
				image_2 = data[i,1]  #  image_2 = data[i,1,:,:]

				if Alg_name == 'AVS_DS_Color_HC_' + str(threshold):
					result[i] = AVS_DS_Color_1(image_1, image_2)
				elif Alg_name == 'AVS_DS_Color_HC_AC_' + str(threshold):
					result[i] = AVS_DS_Color_2(image_1, image_2)
				elif Alg_name == 'AVS_DS_Color_Tangch_' + str(threshold):
					result[i] = AVS_DS_Color_3(image_1, image_2)
				print(i)
			correct = 0
			for i in range(sample_num):
				if np.argmax(result[i]) == np.argmax(label[i]): # (100, 35, 24, 65, 78, 90, 21, 3)     #看哪个细胞激活最多
					correct = correct + 1
				else:
					print(i)
					print(label[i])
					print(result[i])
					#  plt.imshow(data[i,0])
					#  plt.show()
					#  plt.imshow(data[i,1])
					#  plt.show()
					#print(result)
			print(correct)
			acc_object = correct/sample_num
			print(acc_object)
			result_excel[Num_dataset, 0] = correct
			result_excel[Num_dataset, 1] = sample_num
			result_excel[Num_dataset, 2] = acc_object
			Num_dataset = Num_dataset + 1
		index = np.array(index)
		result_excel = pd.DataFrame(result_excel, index = index)
		result_excel.to_excel(writer, sheet_name = str(1), index = True, header=['correct' , 'total' , 'accuracy'])
		writer.save()
