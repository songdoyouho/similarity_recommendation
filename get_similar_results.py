import time
import json
import csv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# 目前使用的類別
class_map_dict = {'hoodie':1,
'polo':2,
'shirt':3,
'sweater':4,
'tank':5,
'tee':6,
'blouse':7,
'top_women_other':8,
'two-piece_upper':9,
'vest_coat':10,
'jacket_blazer':11,
'jacket_cardigan':12,
'jacket_denim':13,
'jacket_poncho':14,
'jacket_windbreaker':15,
'jumpsuit':16,
'romper':17,
'dress_halter':18,
'dungarees':19,
'dress_strap':20,
'two-piece_suit':21,
'jeans':22,
'jeans_long_pants':23,
'jeans_short_pants':24,
'leggings':25,
'slacks':26,
'casual_long_pants':27,
'casual_short_pants':28,
'sports_long_pants':29,
'sports_short_pants':30,
'pants_palazzo':31,
'pants_harem':32,
'culottes':33,
'skirt_layered':34,
'skirt_pleat':35,
'skirt_wrap':36,
'skirt_jeans':37,
'skirt_cargo':38,
'skirt_func':39,
'veil':40,
'invisible':41,
'dress':42,
'top':43}

start_time = time.time()

# 讀要被檢查的圖片的 feature map 
with open('feature_map_results.json', 'r') as iii: # momo_cloth_inference_results
    inference_results = json.load(iii)
    iii.close()
# 讀整個資料庫的 feature map
with open('feature_map_results.json', 'r') as mmm:
    main_results = json.load(mmm)
    mmm.close()

item_features_dict = {}
category_feature_map_list = [[] for i in range(78)]

# 讀答案，得到圖片的屬性及子款式
with open('answer.csv','r',encoding='big5',newline='') as csvfile:
    rrr = csv.reader(csvfile)

    all_rows = []
    for row in rrr:
        all_rows.append(row)

        item_features_dict[row[0][:-4]] = row[1:] # 圖片對應屬性
        #print(row[1:], row[-2])

# 組合 圖片, feature map, 子款式
# category_feature_map_list 的大小是子款式類別大小，使用 index 可以拿到屬於該類子款式的 [圖片名稱, feature map]
new_class_map_dict = {}
class_counter = 0
for key in main_results.keys():
    other_key = key[:-4]
    # 在這邊把性別加進去類別裡面
    keys = new_class_map_dict.keys()
    if item_features_dict[other_key][-1]+'_'+item_features_dict[other_key][-2] not in keys:
        new_class_map_dict[item_features_dict[other_key][-1]+'_'+item_features_dict[other_key][-2]] = class_counter
        class_counter += 1

    category_feature_map_list[new_class_map_dict[item_features_dict[other_key][-1]+'_'+item_features_dict[other_key][-2]]].append([key, main_results[key]])
    #print(item_features_dict[other_key][-1]+'_'+item_features_dict[other_key][-2])

counter = 0
for category_feature_map in category_feature_map_list:
    #print(len(category_feature_map))
    counter += len(category_feature_map)

with open('gender_sub_category_alternatives.json', 'r') as mmm:
    subcategory_alternatives = json.load(mmm)
    mmm.close()

for category_feature_map_index in range(len(category_feature_map_list)):
    if len(category_feature_map_list[category_feature_map_index]) < 40:
        for key in new_class_map_dict:
            if new_class_map_dict[key] == category_feature_map_index:
                #print("< 40 category:", key)
                # 查表補過去
                alternatives = subcategory_alternatives[key]

                for alternative in alternatives:
                    alternative_index = new_class_map_dict[alternative]
                    category_feature_map_list[category_feature_map_index] = category_feature_map_list[category_feature_map_index] + category_feature_map_list[alternative_index]

'''
for category_feature_map_index in range(len(category_feature_map_list)):
    if len(category_feature_map_list[category_feature_map_index]) < 40:
        for key in new_class_map_dict:
            if new_class_map_dict[key] == category_feature_map_index:
                print("< 40 category:", key)
            else:
                print('pass~')

print(counter)
print(new_class_map_dict)
'''

# 拿到每個類別的 feature map，把 feature map 轉換到 gpu 上
gpu_feature_map = []
for category_index in range(len(category_feature_map_list)):
    img_name_and_feature_map_list = category_feature_map_list[category_index]
    target_class_feature_map = []
    for img_name_and_feature_map in img_name_and_feature_map_list:
        target_class_feature_map.append(img_name_and_feature_map[1])
    target_class_feature_map = tf.convert_to_tensor(target_class_feature_map, dtype=tf.float32)
    gpu_feature_map.append(target_class_feature_map)
 
# re-assemble the data，把檔名跟 feature map 拿出來
main_results_key = []
main_results_value = []
for key, value in main_results.items():
    main_results_key.append(key)
    main_results_value.append(value)

inference_results_key = []
inference_results_value = []
for key, value in inference_results.items():
    inference_results_key.append(key)
    inference_results_value.append(value)

print('parsing done!')

top40 = open('similarity_recommendation.csv', 'w', newline='')
top40_writer = csv.writer(top40)

# 製作第一行
top40_writer.writerow(['item_code', 'sort', 'similar_item_code'])  

# loop the validation set
for index in range(len(inference_results_value)):
    print('Processing number: ', index, ' ------------------------------------------------------')

    # 拿出要比對的圖片的 feature map
    inference_result = inference_results_value[index]
    inference_result = tf.convert_to_tensor(inference_result, dtype=tf.float32)

    # 拿到原始圖片的名子
    #answer_list = compare_answer[inference_results_key[index]]
    check_image_name = inference_results_key[index][:-4]
    
    print('check_image_name: ', check_image_name, item_features_dict[check_image_name])

    # 拿到該類別的 feature map
    img_name_and_feature_map_list = category_feature_map_list[new_class_map_dict[item_features_dict[check_image_name][-1]+'_'+item_features_dict[check_image_name][-2]]] # 在這邊把性別加進去類別裡面
    target_class_feature_map = gpu_feature_map[new_class_map_dict[item_features_dict[check_image_name][-1]+'_'+item_features_dict[check_image_name][-2]]]
    
    # 計算 cos 距離
    cos_distance = tf.keras.losses.cosine_similarity(inference_result, target_class_feature_map).numpy()

    cos_min_index_list = []
    cos_mapped_img_list_top5 = []

    li=[]
    for i in range(len(cos_distance)):
        li.append([cos_distance[i],i])
    li.sort()
    sort_index_list = []
    
    for x in li:
        sort_index_list.append(x[1])
    
    output_similarity_counter = 0
    for sort_index in sort_index_list:
        if inference_results_key[index][:-4] == img_name_and_feature_map_list[sort_index][0][:-4]:
            continue

        cos_min_index_list.append(sort_index)
        cos_mapped_img_list_top5.append(img_name_and_feature_map_list[sort_index][0])

        if output_similarity_counter == 4:
            break
        else:
            output_similarity_counter += 1

    print('cos mapped images: ', cos_mapped_img_list_top5)

    output_similarity_counter = 1
    for sort_index in sort_index_list:
        if inference_results_key[index][:-4] == img_name_and_feature_map_list[sort_index][0][:-4]:
            continue

        cos_mapped_img_list_top40 = []
        cos_mapped_img_list_top40.append(inference_results_key[index][:-4])
        cos_mapped_img_list_top40.append(str(output_similarity_counter))
        cos_mapped_img_list_top40.append(img_name_and_feature_map_list[sort_index][0][:-4])

        if output_similarity_counter == 41:
            break
        else:
            output_similarity_counter += 1

        top40_writer.writerow(cos_mapped_img_list_top40)

end_time = time.time() - start_time
print('total time:', end_time)
print('total items:', len(inference_results_value))