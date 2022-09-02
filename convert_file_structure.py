import os
import json
import shutil

input_img_folder = 'main_img'
input_attr_folder = 'momo_attr'
output_for_feature_map = input_img_folder + '_for_feature_map'
os.makedirs(output_for_feature_map, exist_ok=True)

folder_list = os.listdir(input_attr_folder)

no_json_files = []
no_attributes_files = []
no_main_img_files = []
sub_category_is_invisible = []
pass_files = []

pass_count = 0
for folder in folder_list:
    print(folder)
    # 如果 json 檔存在的話
    if os.path.exists(input_attr_folder+'/'+folder+'/attributes.json'):

        # 讀 json 檔看看有沒有屬性結果
        f = open(input_attr_folder+'/'+folder+'/attributes.json', 'r')
        data = json.load(f)
        # 如果有結果，會通過
        try:
            code = data['result']['detect_status']['code']
        except:
            no_attributes_files.append(folder)
            continue

        if data['result']['detect_status']['code'] != 1:
            no_attributes_files.append(folder)
            continue
        
        # 看看跑出來的子款式是不是 “invisible”，不是才通過
        invisible_flag = False
        for key in data['result']['cloth_attr']:
                sub_category = data['result']['cloth_attr'][key]['sub_category']
                if sub_category == 'invisible':
                    print('--------------------------------------', sub_category)
                    sub_category_is_invisible.append(folder)
                    invisible_flag = True
        
        if invisible_flag:
            continue
        
        # 檢查放圖片的資料夾是否存在
        if os.path.exists(input_img_folder+'/'+folder):
            pass_count += 1
            pass_files.append(folder)
            img_list = os.listdir(input_img_folder+'/'+folder)

            for img_name in img_list:
                shutil.copyfile(input_img_folder+'/'+folder+'/'+img_name, output_for_feature_map+'/'+folder+'.jpg')
        else:
            no_main_img_files.append(folder)

    else:
        no_json_files.append(folder)


print('no_json_files:', no_json_files)
print('no_attributes_files:', no_attributes_files)
print('no_main_img_files:', no_main_img_files)
print('sub_category_is_invisible:', sub_category_is_invisible)
print(len(no_json_files), len(no_attributes_files), len(no_main_img_files), len(folder_list), len(sub_category_is_invisible), pass_count)

output_json = {'no_json_files':no_json_files, 'no_attributes_files':no_attributes_files, 'no_main_img_files':no_main_img_files, 'sub_category_is_invisible':sub_category_is_invisible, 'pass_files':pass_files}
with open("cleaned_data_result.json", "w") as outfile:
    json.dump(output_json, outfile)

os.makedirs('momo_no_json_file', exist_ok=True)
for item in no_json_files:
    shutil.move(input_attr_folder+'/'+item, 'momo_no_json_file/'+item)

os.makedirs('momo_empty_attribute', exist_ok=True)
for item in no_attributes_files:
    shutil.move(input_attr_folder+'/'+item, 'momo_empty_attribute/'+item)

os.makedirs('momo_no_img_file', exist_ok=True)
for item in no_main_img_files:
    shutil.move(input_attr_folder+'/'+item, 'momo_no_img_file/'+item)

os.makedirs('momo_attr_subcategory_invisible', exist_ok=True)
for item in sub_category_is_invisible:
    shutil.move(input_attr_folder+'/'+item, 'momo_attr_subcategory_invisible/'+item)