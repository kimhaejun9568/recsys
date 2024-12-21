def clean_review(review):
    if not isinstance(review, str):  # 문자열이 아닌 경우
        return ''
    review = review.lower()
    for p in punctuations:
        review = review.replace(p, ' ')
    tokens = WordPunctTokenizer().tokenize(review)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from collections import defaultdict
from PIL import Image
import os
from nltk.tokenize import WordPunctTokenizer

# Config, Logger, Model, Dataset 클래스 등은 기존 코드에서 import하거나 복사해온다고 가정
# IntegratedModel, Config, clean_review, sentiment_analysis 등 기존 코드에서 정의한 함수/클래스를 재활용

# 1. Config와 모델 로드
config = Config()
model = IntegratedModel(config, clip_model, text_model).to(config.device)
model.load_state_dict(torch.load('data/yelp/best_model.pth'))
model.eval()

# 2. 새로운 CSV 로딩 (new_data.csv)
new_data_path = 'data/yelp/restaurant_reviews.csv'
df_new = pd.read_csv(new_data_path)

# 전처리 (clean_review) 적용
df_new['review'] = df_new['review'].apply(clean_review)
#df_new = df_new.dropna(subset=['review'])  # NaN 값 제거

# 3. photos.json 로드
photo_json = os.path.join(config.data_dir,'kakaophotos.json')
photos_df = pd.read_json(photo_json, orient='records', lines=True)
photo_groups = defaultdict(lambda: defaultdict(list))
for row in photos_df.itertuples():
    bid = row.business_id
    lbl = row.label
    pid = row.photo_id
    if lbl in config.views:
        photo_groups[bid][lbl].append(pid)

# 4. Dataset, DataLoader 준비
class InferenceDataset(Dataset):
    def __init__(self, df, photo_groups, config):
        self.df=df.reset_index(drop=True)
        self.photo_groups=photo_groups
        self.config=config
        def split_to_sentences(text):
            sents=text.split('.')
            sents=[s.strip() for s in sents if len(s.strip().split())>5]
            return sents
        self.df['sentences']=self.df['review'].apply(split_to_sentences)
        self.retain_idx = [len(x)>0 for x in self.df['sentences']]
        self.df = self.df[self.retain_idx].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        userID=self.df.loc[idx,'userID']
        itemID=self.df.loc[idx,'itemID']
        rating=self.df.loc[idx,'rating']  # 예측 시 실제 rating이 없어도 0 등의 더미값
        sentences=self.df.loc[idx,'sentences']
        ui_sentences=sentences[:self.config.max_ui_sent_count]
        views_img_paths=[]
        for v in self.config.views:
            pids = self.photo_groups[itemID].get(v,[])
            if len(pids)==0:
                views_img_paths.append(['unknown.jpg'])
            else:
                chosen_pid=random.choice(pids)
                views_img_paths.append([os.path.join(self.config.data_dir, 'restaurant_photos', chosen_pid+'.jpg')])
        return {
            'userID':userID,
            'itemID':itemID,
            'rating':rating,
            'sentences':sentences,
            'ui_sentences':ui_sentences,
            'views_img_paths':views_img_paths
        }

def collate_fn(batch):
    ratings = torch.tensor([b['rating'] for b in batch], dtype=torch.float)
    all_sentences=[b['sentences'] for b in batch]
    all_ui_sentences=[b['ui_sentences'] for b in batch]
    all_views_paths=[b['views_img_paths'] for b in batch]
    userIDs=[b['userID'] for b in batch]
    itemIDs=[b['itemID'] for b in batch]
    return (all_sentences, all_ui_sentences, all_views_paths, ratings, userIDs, itemIDs)

inference_data = InferenceDataset(df_new, photo_groups, config)
inference_loader = DataLoader(inference_data, batch_size=config.batch_size, collate_fn=collate_fn)

# 5. 예측 실행
predictions = []
with torch.no_grad():
    for batch in inference_loader:
        (all_sentences, all_ui_sentences, all_views_paths, ratings, userIDs, itemIDs) = batch
        ratings = ratings.to(config.device)
        pred, loss = model(all_sentences, all_ui_sentences, all_views_paths, ratings)
        predictions.extend(pred.cpu().numpy().tolist())

# predictions 리스트 길이와 df_new 길이 비교
print(f"Length of df_new: {len(df_new)}")
print(f"Length of predictions: {len(predictions)}")

# df_new를 Dataset에서 retain된 인덱스로 필터링
df_new_filtered = df_new.iloc[inference_data.retain_idx].reset_index(drop=True)

# predictions 추가
df_new_filtered['predicted_rating'] = predictions

# 저장
df_new_filtered.to_csv('data/yelp/new_data_predicted.csv', index=False)
print("Inference complete. Results saved to data/yelp/new_data_predicted.csv")
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. predictions 리스트 길이와 df_new 길이 비교
print(f"Length of df_new: {len(df_new)}")
print(f"Length of predictions: {len(predictions)}")

# 2. df_new를 Dataset에서 retain된 인덱스로 필터링
df_new_filtered = df_new.iloc[inference_data.retain_idx].reset_index(drop=True)

# 3. predictions 추가
df_new_filtered['predicted_rating'] = predictions

# 4. 정답값과 비교
true_ratings = df_new_filtered['rating'].values  # 실제 정답값
predicted_ratings = df_new_filtered['predicted_rating'].values  # 모델 예측값

# MSE 계산
mse = mean_squared_error(true_ratings, predicted_ratings)
mae = mean_absolute_error(true_ratings, predicted_ratings)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# 5. 결과 저장
df_new_filtered.to_csv('data/yelp/new_data_predicted.csv', index=False)
print("Inference complete. Results saved to data/yelp/new_data_predicted.csv")
