import csv
import time
import json

from recommend.recommend_engine import RecommendEngine
from test.utils import load_json

class ToolBox:
    def __init__(self) -> None:
        """ 初始化一些檔案 """
        f = open('clothes_recommend/feature_map_results.json')
        self.momo_feature_map_dict = json.load(f)
        f.close()

        self.momo_recommend_engine = self.create_engine('/home/training/fashion_recommend_api/clothes_recommend/momo_attr',
                                    '/home/training/fashion_recommend_api/recommend/attribute_20220401.json')

    def create_engine(self, dataset_path: str, weight_path: str) -> object:
        """ 初始化柏宣的推薦引擎
        parameters:
            dataset_path: 資料集資料夾所放的路徑，裡面含有每個商品的屬性 json 檔
            weight_path: 屬性推薦引擎所需要的 weight file，描述每個屬性的重要程度
        
        return:
            recommend_engine: 初始化好的物件
        """
        recommend_engine = RecommendEngine(dataset_path, weight_path)
        recommend_engine.update()
        print('engine created !!!!!!!!!!!')
        
        return recommend_engine

if __name__ == '__main__':
    ToolBox = ToolBox()

    # 拿 沒有圖片但是有屬性的結果出來做
    json_reader = open('similarity_recommendation/cleaned_data_result.json')
    cleaned_data_result = json.load(json_reader)

    with open('similarity_recommendation/similarity_recommendation.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item_id in cleaned_data_result['no_main_img_files']:
            #print('item_id:', item_id)
            json_path = '/home/training/fashion_recommend_api/clothes_recommend/momo_attr/' + item_id + '/attributes.json'
            clothes_dict = load_json(str(json_path))
            recommend_results = ToolBox.momo_recommend_engine.recommend(clothes_dict)
            #print('recommend_results:', recommend_results)

            # 把結果塞到 similarity_recommendation.csv 後面
            if len(recommend_results['similar']) == 40:
                output_counter = 1
                for tmp_recommend_id in recommend_results['similar']:
                    recommend_id = tmp_recommend_id['id']
                    output_line = [str(item_id), str(output_counter), str(recommend_id)]
                    print(output_line)
                    output_counter += 1
                    writer.writerow(output_line)
