import os
import json
import csv

class_dict = {'hoodie':1,
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

input_attr_folder = 'momo_attr'

topk = open('topk_ids.csv', 'w', newline='')
topk_writer = csv.writer(topk)
topk_writer.writerow(['img_name', '1'])
answer = open('answer.csv', 'w', newline='')
answer_writer = csv.writer(answer)

json_list = os.listdir(input_attr_folder)

for json_file in json_list:
    f = open(input_attr_folder + '/' + json_file + '/attributes.json')
    data = json.load(f)

    try:
        tmp = data['result']['detect_status']['code']
    except:
        continue

    if data['result']['detect_status']['code'] == 1:
        for key in data['result']['cloth_attr'].keys():
            print(data)

            if key == 'upper':
                print('-------------------------------- upper')
                print(data['result']['cloth_attr']['upper']['coat']) # 衣長
                print(data['result']['cloth_attr']['upper']['color'][0].strip(' 100%')) # 顏色
                
                for neck_key in data['result']['cloth_attr']['upper']['neckband'].keys():
                    print(neck_key) # 領口款式
                    neckband = neck_key

                print(data['result']['cloth_attr']['upper']['pattern_size']) # 圖案大小
                print(data['result']['cloth_attr']['upper']['sleeve']) # 袖長
                print(data['result']['cloth_attr']['upper']['sub_category']) # 服飾子款式
                #print(data['result']['cloth_loc']['upper']) # bbox 位置
                print(json_file)

                topk_writer.writerow([json_file + '.jpg', class_dict[data['result']['cloth_attr']['upper']['sub_category']]])
                answer_writer.writerow([json_file + '.jpg', 'upper', data['result']['cloth_attr']['upper']['coat'], data['result']['cloth_attr']['upper']['color'][0].strip(' 100%'), neckband, data['result']['cloth_attr']['upper']['pattern_size'], data['result']['cloth_attr']['upper']['sleeve'], data['result']['cloth_attr']['upper']['sub_category'], data['request']['gender']])

            if key == 'full':
                print('******************************** full')
                print(data['result']['cloth_attr']['full']['coat']) # 衣長
                print(data['result']['cloth_attr']['full']['color'][0].strip(' 100%')) # 顏色
                
                for neck_key in data['result']['cloth_attr']['full']['neckband'].keys():
                    print(neck_key) # 領口款式
                    neckband = neck_key

                print(data['result']['cloth_attr']['full']['pattern_size']) # 圖案大小
                print(data['result']['cloth_attr']['full']['sleeve']) # 袖長
                print(data['result']['cloth_attr']['full']['sub_category']) # 服飾子款式
                #print(data['result']['cloth_loc']['full']) # bbox 位置
                print(json_file)

                topk_writer.writerow([json_file + '.jpg', class_dict[data['result']['cloth_attr']['full']['sub_category']]])
                answer_writer.writerow([json_file + '.jpg', 'full', data['result']['cloth_attr']['full']['coat'], data['result']['cloth_attr']['full']['color'][0].strip(' 100%'), neckband, data['result']['cloth_attr']['full']['pattern_size'], data['result']['cloth_attr']['full']['sleeve'], data['result']['cloth_attr']['full']['sub_category'], data['request']['gender']])
                
            if key == 'lower':
                print('++++++++++++++++++++++++++++++++ lower')
                for bottom_key in data['result']['cloth_attr']['lower']['bottom'].keys():
                    print(data['result']['cloth_attr']['lower']['bottom'][bottom_key]) # 下身長度
                    bottom_length = data['result']['cloth_attr']['lower']['bottom'][bottom_key]

                print(data['result']['cloth_attr']['lower']['shape']) # 下身形狀
                print(data['result']['cloth_attr']['lower']['color'][0].strip(' 100%')) # 顏色
                print(data['result']['cloth_attr']['lower']['pattern_size']) # 圖案大小
                print(data['result']['cloth_attr']['lower']['sub_category']) # 服飾子款式
                #print(data['result']['cloth_loc']['lower']) # bbox 位置
                print(json_file)

                topk_writer.writerow([json_file + '.jpg', class_dict[data['result']['cloth_attr']['lower']['sub_category']]])
                answer_writer.writerow([json_file + '.jpg', 'lower', bottom_length, data['result']['cloth_attr']['lower']['shape'], data['result']['cloth_attr']['lower']['color'][0].strip(' 100%'), data['result']['cloth_attr']['lower']['pattern_size'], data['result']['cloth_attr']['lower']['sub_category'], data['request']['gender']])
