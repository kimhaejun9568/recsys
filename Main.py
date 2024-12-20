import os
import sys
import json
import random
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import nltk
nltk.download('punkt')
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

business_input = 'yelp/yelp_academic_dataset_business.json'
review_input = 'yelp/yelp_academic_dataset_review.json'
photo_input = 'yelp/photos.json'
save_dir = 'data/yelp'
os.makedirs(save_dir, exist_ok=True)

restaurants = {}
with open(business_input,'r',encoding='utf-8') as f:
    for line in f:
        biz = json.loads(line.strip())
        cats = biz.get('categories','')
        if cats and 'Restaurants' in cats:
            restaurants[biz['business_id']] = biz
print(f"Total Restaurants: {len(restaurants)}")

photo_df = []
with open(photo_input, 'r', encoding='utf-8') as f:
    for line in f:
        p = json.loads(line.strip())
        if p['label'] in ['food', 'drink', 'inside', 'outside']:
            photo_df.append(p)
photo_df = pd.DataFrame(photo_df)
grouped = photo_df.groupby('business_id')['label'].apply(list)
def valid_views(labels):
    return (
        labels.count('inside') >= 1 and
        'food' in labels and
        'drink' in labels and
        'outside' in labels
    )
valid_biz = [biz_id for biz_id, labels in grouped.items() if valid_views(labels)]
print(f"Business with all 4 views (and 'inside' >= 1): {len(valid_biz)}")
valid_biz = set(valid_biz).intersection(restaurants.keys())
print(f"Restaurants meeting all view conditions: {len(valid_biz)}")
photo_df = photo_df[photo_df['business_id'].isin(valid_biz)]
total_photos = photo_df.shape[0]
print(f"Total photos used: {total_photos}")
indoor_outdoor_photos = photo_df[photo_df['label'].isin(['inside', 'outside'])].shape[0]
print(f"Total 'inside' and 'outside' photos: {indoor_outdoor_photos}")

user_count=defaultdict(int)
biz_count=defaultdict(int)
with open(review_input,'r',encoding='utf-8') as f:
    for line in f:
        rv=json.loads(line.strip())
        if rv['business_id'] in valid_biz:
            user_count[rv['user_id']] += 1
            biz_count[rv['business_id']] += 1
final_users = {u for u,c in user_count.items() if c>=5}
final_biz = {b for b,c in biz_count.items() if c>=5}
final_biz = final_biz.intersection(valid_biz)
print(f"After 5-cores filtering: {len(final_biz)} businesses")
filtered_reviews=[]
with open(review_input,'r',encoding='utf-8') as f:
    for line in f:
        rv=json.loads(line.strip())
        if rv['business_id'] in final_biz and rv['user_id'] in final_users:
            filtered_reviews.append({
                'user_id':rv['user_id'],
                'business_id':rv['business_id'],
                'text':rv['text'],
                'stars':rv['stars']
            })
df_reviews = pd.DataFrame(filtered_reviews)
user_count2 = df_reviews.groupby('user_id')['business_id'].count()
biz_count2 = df_reviews.groupby('business_id')['user_id'].count()
final_users2 = user_count2[user_count2>=5].index
final_biz2 = biz_count2[biz_count2>=5].index
df_reviews = df_reviews[df_reviews['user_id'].isin(final_users2) & df_reviews['business_id'].isin(final_biz2)]
print(f"Final: {df_reviews['business_id'].nunique()} businesses, {df_reviews['user_id'].nunique()} users, {len(df_reviews)} reviews")
df_biz = pd.DataFrame([restaurants[b] for b in df_reviews['business_id'].unique()])
df_biz.to_json(os.path.join(save_dir,'business.json'), orient='records', lines=True)
df_photos = photo_df[photo_df['business_id'].isin(df_reviews['business_id'].unique())]
df_photos.to_json(os.path.join(save_dir,'photos.json'), orient='records', lines=True)
df_reviews.to_json(os.path.join(save_dir,'reviews.json'), orient='records', lines=True)
print("Data filtering complete!")

nltk.download('punkt')
with open('embedding/stopwords.txt','r') as f:
    stop_words=set(f.read().splitlines())
with open('embedding/punctuations.txt','r') as f:
    punctuations=set(f.read().splitlines())
    if '.' in punctuations:
        punctuations.remove('.')

def process_dataset(reviews_path, meta_path, save_dir, train_rate, select_cols):
    os.makedirs(save_dir, exist_ok=True)
    data=[]
    with open(reviews_path,'r',encoding='utf-8') as f:
        for line in f:
            item=json.loads(line.strip())
            data.append([item[c] for c in select_cols])
    df=pd.DataFrame(data, columns=['userID','itemID','review','rating'])
    df['user_num']=df.groupby('userID').ngroup()
    df['item_num']=df.groupby('itemID').ngroup()
    with open('embedding/stopwords.txt','r') as f:
        stop_words=set(f.read().splitlines())
    with open('embedding/punctuations.txt','r') as f:
        punctuations=set(f.read().splitlines())
        if '.' in punctuations:
            punctuations.remove('.')
    def clean_review(review):
        review=review.lower()
        for p in punctuations:
            review=review.replace(p,' ')
        review=WordPunctTokenizer().tokenize(review)
        review=[w for w in review if w not in stop_words]
        return ' '.join(review)
    df = df.drop(df[[not isinstance(x,str) or len(x)==0 for x in df['review']]].index)
    df['review']=df['review'].apply(clean_review)
    train, valid = train_test_split(df, test_size=1-train_rate, random_state=3)
    valid, test = train_test_split(valid, test_size=0.5, random_state=4)
    train.to_csv(os.path.join(save_dir,'train.csv'), index=False)
    valid.to_csv(os.path.join(save_dir,'valid.csv'), index=False)
    test.to_csv(os.path.join(save_dir,'test.csv'), index=False)
    print(f"train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

reviews_path = os.path.join(save_dir,'reviews.json')
meta_path = os.path.join(save_dir,'business.json')
process_dataset(reviews_path, meta_path, save_dir, train_rate=0.8, select_cols=['user_id','business_id','text','stars'])
with open('embedding/stopwords.txt','r') as f:
    stop_words=set(f.read().splitlines())
with open('embedding/punctuations.txt','r') as f:
    punctuations=set(f.read().splitlines())
    if '.' in punctuations:
        punctuations.remove('.')
train_path = 'data/yelp/train.csv'
valid_path = 'data/yelp/valid.csv'
test_path = 'data/yelp/test.csv'
train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df = pd.read_csv(test_path)
train_df = train_df.dropna(subset=['review'])
valid_df = valid_df.dropna(subset=['review'])
test_df = test_df.dropna(subset=['review'])
train_df.to_csv(train_path, index=False)
valid_df.to_csv(valid_path, index=False)
test_df.to_csv(test_path, index=False)
print("=== Train NaN Counts ===")
print(train_df.isna().sum())
print("\n=== Valid NaN Counts ===")
print(valid_df.isna().sum())
print("\n=== Test NaN Counts ===")
print(test_df.isna().sum())
print("\nCheck if 'review' column has NaN in train:")
print(train_df['review'].isna().sum())
print("Check if 'review' column has NaN in valid:")
print(valid_df['review'].isna().sum())
print("Check if 'review' column has NaN in test:")
print(test_df['review'].isna().sum())

class Config:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_epochs = 20
        self.batch_size = 4
        self.learning_rate = 1e-5
        self.data_dir = 'data/yelp'
        self.views = ['inside','outside','food','drink']
        self.max_sent_length = 20
        self.max_sent_count = 20
        self.min_sent_count = 5
        self.max_ui_sent_count = 5
        self.threshold = 0.35
        self.loss_v_rate = 0.1
        self.global_pref_dim = 128
        self.hidden_dim = 384
        self.cross_attention_heads = 8
        self.cross_attention_layers = 2
        self.lr_decay = 0.99

config = Config()
def get_logger():
    logging.root.setLevel(0)
    formatter=logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger=logging.getLogger(__name__)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    sh=logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
logger = get_logger()
logger.info(config)

with open('embedding/stopwords.txt','r') as f:
    stop_words=set(f.read().splitlines())
with open('embedding/punctuations.txt','r') as f:
    punctuations=set(f.read().splitlines())
    if '.' in punctuations:
        punctuations.remove('.')

def clean_review(review):
    review=review.lower()
    for p in punctuations:
        review=review.replace(p,' ')
    tokens=WordPunctTokenizer().tokenize(review)
    tokens=[w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

train_path=os.path.join(config.data_dir,'train.csv')
valid_path=os.path.join(config.data_dir,'valid.csv')
test_path=os.path.join(config.data_dir,'test.csv')
photo_json=os.path.join(config.data_dir,'photos.json')
photo_path=os.path.join(config.data_dir,'photos')
df_train = pd.read_csv(train_path)
df_valid = pd.read_csv(valid_path)
df_test = pd.read_csv(test_path)
df_train['review']=df_train['review'].apply(clean_review)
df_valid['review']=df_valid['review'].apply(clean_review)
df_test['review']=df_test['review'].apply(clean_review)
photos_df = pd.read_json(photo_json, orient='records', lines=True)
photo_groups = defaultdict(lambda: defaultdict(list))
for row in photos_df.itertuples():
    bid=row.business_id
    lbl=row.label
    pid=row.photo_id
    if lbl in config.views:
        photo_groups[bid][lbl].append(pid)

class YelpDataset(Dataset):
    def __init__(self, df, photo_groups, config, mode='train'):
        self.df=df.reset_index(drop=True)
        self.photo_groups=photo_groups
        self.config=config
        self.mode=mode
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
        rating=self.df.loc[idx,'rating']
        sentences=self.df.loc[idx,'sentences']
        ui_sentences=sentences[:self.config.max_ui_sent_count]
        views_img_paths=[]
        for v in self.config.views:
            pids = self.photo_groups[itemID].get(v,[])
            if len(pids)==0:
                views_img_paths.append(['unknown.jpg'])
            else:
                chosen_pid=random.choice(pids)
                views_img_paths.append([os.path.join(photo_path, chosen_pid+'.jpg')])
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

train_data=YelpDataset(df_train, photo_groups, config, mode='train')
valid_data=YelpDataset(df_valid, photo_groups, config, mode='valid')
test_data=YelpDataset(df_test, photo_groups, config, mode='test')
train_dlr=DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
valid_dlr=DataLoader(valid_data, batch_size=config.batch_size, collate_fn=collate_fn)
test_dlr=DataLoader(test_data, batch_size=config.batch_size, collate_fn=collate_fn)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(config.device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(config.device)

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.layernorm2=nn.LayerNorm(hidden_dim)
    def forward(self, query, key_value):
        attn_out, _ = self.attn(query, key_value, key_value)
        x = self.layernorm(query+attn_out)
        ff_out = self.ff(x)
        x = self.layernorm2(x+ff_out)
        return x

sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model = sentiment_model.to(config.device)
sentiment_model.eval()

def sentiment_analysis(sentence: str) -> float:
    if not sentence.strip():
        return 0.5
    inputs = sentiment_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(config.device)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        positive_prob = probs[0,1].item()
        return positive_prob

class FiLMGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gamma=nn.Linear(hidden_dim, hidden_dim)
        self.beta=nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, cond):
        gamma=self.gamma(cond)
        beta=self.beta(cond)
        return x*gamma+beta

class IntegratedModel(nn.Module):
    def __init__(self, config, clip_model, text_model):
        super().__init__()
        self.config=config
        self.clip_model=clip_model
        self.text_model=text_model
        self.global_pref=nn.Parameter(torch.randn(config.global_pref_dim))
        self.cross_attention_layers=nn.ModuleList([CrossAttentionBlock(config.hidden_dim, config.cross_attention_heads) for _ in range(config.cross_attention_layers)])
        self.film_gate=FiLMGate(config.hidden_dim)
        self.pred_head=nn.Sequential(nn.Linear(config.hidden_dim,1))
        self.img_map=nn.Linear(512,384)
    def encode_images(self, image_paths_batch):
        batch_img_emb=[]
        for sample in image_paths_batch:
            all_view_emb=[]
            for view_imgs in sample:
                imgs=[]
                for img_path in view_imgs:
                    if not os.path.exists(img_path):
                        img = Image.new('RGB',(224,224),(0,0,0))
                    else:
                        img = Image.open(img_path).convert('RGB')
                    imgs.append(img)
                inputs = clip_processor(images=imgs, return_tensors="pt").to(self.config.device)
                with torch.no_grad():
                    img_emb=self.clip_model.get_image_features(**inputs)
                view_emb=img_emb.mean(dim=0,keepdim=True)
                all_view_emb.append(view_emb)
            sample_emb = torch.cat(all_view_emb, dim=0).mean(dim=0,keepdim=True)
            sample_emb=self.img_map(sample_emb)
            sample_emb=sample_emb/ sample_emb.norm(dim=-1,keepdim=True)
            batch_img_emb.append(sample_emb)
        batch_img_emb=torch.cat(batch_img_emb,dim=0)
        return batch_img_emb
    def encode_text(self, sentences_batch):
        all_emb=[]
        for sents in sentences_batch:
            if len(sents)==0:
                sents=["empty"]
            emb = self.text_model.encode(sents, convert_to_tensor=True, device=self.config.device)
            emb = emb.mean(dim=0,keepdim=True)
            emb=emb/emb.norm(dim=-1,keepdim=True)
            all_emb.append(emb)
        all_emb=torch.cat(all_emb,dim=0)
        return all_emb
    def forward(self, all_sentences, all_ui_sentences, all_views_paths, ratings):
        text_emb=self.encode_text(all_sentences)
        ui_text_emb=self.encode_text(all_ui_sentences)
        sentiment_scores=[]
        for ui_sents in all_ui_sentences:
            if len(ui_sents)==0:
                sentiment_scores.append(0.5)
            else:
                scs=[sentiment_analysis(s) for s in ui_sents]
                sentiment_scores.append(sum(scs)/len(scs))
        sentiment_scores=torch.tensor(sentiment_scores, device=self.config.device).unsqueeze(-1)
        img_emb=self.encode_images(all_views_paths)
        query=text_emb.unsqueeze(1)
        key_value=img_emb.unsqueeze(1)
        for layer in self.cross_attention_layers:
            query=layer(query,key_value)
        fused_emb=query.squeeze(1)
        global_pref=self.global_pref.to(self.config.device).unsqueeze(0).expand(fused_emb.size(0),-1)
        cond=torch.cat([global_pref, sentiment_scores],dim=-1)
        cond_proj=nn.Linear(self.config.global_pref_dim+1, self.config.hidden_dim).to(self.config.device)
        cond_vec=cond_proj(cond)
        gated_emb=self.film_gate(fused_emb,cond_vec)
        pred=self.pred_head(gated_emb).squeeze(-1)
        loss=F.mse_loss(pred,ratings)
        return pred, loss

model=IntegratedModel(config, clip_model, text_model).to(config.device)
optimizer=torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)

def evaluate(dlr):
    model.eval()
    mse_total = 0
    count = 0
    with torch.no_grad():
        for batch in dlr:
            (all_sentences, all_ui_sentences, all_views_paths, ratings, userIDs, itemIDs) = batch
            ratings = ratings.to(config.device)
            pred, loss = model(all_sentences, all_ui_sentences, all_views_paths, ratings)
            mse_total += loss.item() * len(pred)
            count += len(pred)
    return mse_total / count

best_val_mse = float('inf')
for epoch in range(config.train_epochs):
    print(f"Starting epoch {epoch + 1}/{config.train_epochs}")
    model.train()
    total_loss = 0
    total_samples = 0
    batch_loss = 0
    batch_samples = 0
    batch_count = 0
    for i, batch in enumerate(tqdm(train_dlr, desc=f"Training epoch {epoch + 1}")):
        (all_sentences, all_ui_sentences, all_views_paths, ratings, userIDs, itemIDs) = batch
        ratings = ratings.to(config.device)
        optimizer.zero_grad()
        pred, loss = model(all_sentences, all_ui_sentences, all_views_paths, ratings)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item() * len(pred)
        batch_samples += len(pred)
        total_loss += loss.item() * len(pred)
        total_samples += len(pred)
        batch_count += 1
        if batch_count == 10000:
            batch_mse = batch_loss / batch_samples
            logger.info(f"Epoch {epoch + 1}, Batch {i + 1}: Current Batch MSE = {batch_mse:.4f}")
            batch_loss = 0
            batch_samples = 0
            batch_count = 0
    scheduler.step()
    train_mse = total_loss / total_samples
    val_mse = evaluate(valid_dlr)
    logger.info(f"Epoch {epoch + 1}: Train MSE = {train_mse:.4f}, Validation MSE = {val_mse:.4f}")
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        torch.save(model.state_dict(),'data/yelp/best_model.pth' )
        print(f"New best model saved with Validation MSE = {best_val_mse:.4f}")

test_mse = evaluate(test_dlr)
logger.info(f"Test MSE = {test_mse:.4f}")
