#!/bin/bash

# 執行清理資料集
python convert_file_structure.py

# 執行獲得每個商品的子款式結果
python produce_topkids_and_answer_16_wan.py

# 執行獲得 feature map json 檔案
python inference_siamese_network.py

# 執行獲得有圖片有屬性的相似推薦結果
python get_similar_results.py

# 執行獲得沒有圖片有屬性的相似推薦結果
python get_similar_results_without_img.py