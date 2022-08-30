#!/bin/bash


# 執行清理資料集
python convert_file_structure.py

# 執行獲得每個商品的子款式結果
python produce_topkids_and_answer_16_wan.py

# 執行獲得 feature map json 檔案
python get_similar_results.py