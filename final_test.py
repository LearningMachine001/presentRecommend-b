import os
import re
import uuid
import torch
import pandas as pd
from collections import defaultdict
from keybert import KeyBERT
from konlpy.tag import Okt
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification, AutoTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
import torch.nn as nn

# 모델 로드
model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name)
interest_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
interest_model.load_state_dict(torch.load("./kobert_importance.pth", map_location="cpu"))
interest_model.eval()

kw_model = KeyBERT(model="distiluse-base-multilingual-cased-v1")
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
okt = Okt()

class KoBertExtendedModel(nn.Module):
    def __init__(self, model_name="skt/kobert-base-v1", num_subjects=20):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.score_head = nn.Linear(768, 1)
        self.awkward_head = nn.Linear(768, 2)
        self.subject_head = nn.Linear(768, num_subjects)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        score = self.score_head(pooled_output)
        awkward = self.awkward_head(pooled_output)
        subject = self.subject_head(pooled_output)
        return score, awkward, subject

topic_tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)
topic_model = KoBertExtendedModel()
topic_model.load_state_dict(torch.load("kobert_extended_with_subject.pth", map_location="cpu"), strict=False)
topic_model.eval()

subject_id2name = {0:"미용",1:"스포츠/레저",2:"교육",3:"가족",5:"영화/만화",6:"교통",7:"여행",
                   8:"회사/아르바이트",9:"건강",10:"연애/결혼",11:"게임",12:"계절/날씨",13:"방송/연예",
                   14:"사회이슈",15:"주거와 생활",16:"반려동물",17:"군대",18:"식음료"}

subject_to_main_category = {0:"뷰티",1:"레저/스포츠",2:"리빙/도서",3:"디지털/가전",5:"패션",
                            6:"디지털/가전",7:"레저/스포츠",8:"리빙/도서",9:"식품",10:"패션",
                            11:"디지털/가전",12:"식품",13:"패션",14:"리빙/도서",
                            15:"리빙/도서",16:"유아동/반려",17:"식품",18:"식품"}

category_to_file = {
    "뷰티": "category_files/beauty.csv",
    "레저/스포츠": "category_files/sport.csv",
    "리빙/도서": "category_files/living.csv",
    "디지털/가전": "category_files/digital.csv",
    "패션": "category_files/fashion.csv",
    "식품": "category_files/food.csv",
    "유아동/반려": "category_files/baby.csv"
}

os.makedirs("cached_embeddings", exist_ok=True)

with open("stopwords-ko.txt", encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())

def load_or_build_embeddings(category: str, intimacy_score: float):
    csv_path = category_to_file.get(category)
    if not csv_path:
        return []
    if intimacy_score < 2:
        suffix = "_2"
    elif intimacy_score < 3:
        suffix = "_3"
    elif intimacy_score < 4:
        suffix = "_4"
    else:
        suffix = "_5"

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    cache_path = f"cached_embeddings/{base_name}{suffix}.pt"

    if os.path.exists(cache_path):
        return torch.load(cache_path)

    fallback_path = f"cached_embeddings/{base_name}.pt"
    if os.path.exists(fallback_path):
        print(f"[!] {cache_path} 없음 → {fallback_path} 사용")
        return torch.load(fallback_path)

    print(f"[!] 캐시 없음: {cache_path} / {fallback_path}")
    return []

def classify_interest_batch(sentences):
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        logits = interest_model(**inputs).logits
        labels = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    return labels.tolist()

def classify_topic(sentence):
    inputs = topic_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        _, _, subject_logits = topic_model(inputs["input_ids"], inputs["attention_mask"])
        subject_id = torch.argmax(subject_logits, dim=1).item()
    return subject_id2name.get(subject_id, "알 수 없음"), subject_to_main_category.get(subject_id, "없음")

def classify_avg_score_from_pairs(messages):
    if len(messages) < 2:
        return 0.0
    pairs = [(messages[i], messages[i + 1]) for i in range(len(messages) - 1)]
    scores = []
    for a, b in pairs:
        input_text = a.strip() + " [SEP] " + b.strip()
        inputs = topic_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            score, _, _ = topic_model(inputs["input_ids"], inputs["attention_mask"])
            score_value = torch.sigmoid(score).item() * 8
            scores.append(score_value)
    return round(sum(scores) / len(scores), 2) if scores else 0.0

def extract_kakao_dialogues(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    data_by_date = defaultdict(list)
    for line in lines:
        if m := re.search(r"(\d{4})년 (\d{1,2})월 (\d{1,2})일", line):
            y, m_, d = m.groups()
            current_date = f"{int(y):04d}-{int(m_):02d}-{int(d):02d}"
        elif re.search(r"[오전|오후]+\s*\d{1,2}:\d{2},\s*[^:]+:", line):
            msg = re.sub(r"^\d{4}\. \d{1,2}\. \d{1,2}\. [오전|오후]+\s*\d{1,2}:\d{2},\s*[^:]+:\s*", "", line).strip()
            if msg:
                data_by_date[current_date].append(msg)
    return data_by_date

def is_valid_conversation(msg):
    return bool(re.search(r"[가-힣]", msg)) and not re.search(r"https?://|총\s*금액", msg)

def extract_interest_weighted_keywords(sentences):
    keyword_scores = defaultdict(float)
    labels = classify_interest_batch(sentences)
    for sentence, label in zip(sentences, labels):
        nouns = {n for n in okt.nouns(sentence) if n not in stopwords and len(n) > 1}
        for kw, score in kw_model.extract_keywords(sentence, (1, 2), None, top_n=5):
            if all(tok in nouns for tok in kw.split()):
                multiplier = 2.5 if " " in kw else 2.0
                keyword_scores[kw] += score * (multiplier if label == 1 else 0.5)
        for noun in nouns:
            keyword_scores[noun] += 0.3 if label == 1 else 0.1

    filtered = [(k, v) for k, v in keyword_scores.items() if all(not re.search(r"(다|어|지|음)$", t) for t in k.split())]
    return sorted(filtered, key=lambda x: x[1], reverse=True)

def recommend_products_from_keywords(sorted_keywords, allowed_category, intimacy_score):
    products = load_or_build_embeddings(allowed_category, intimacy_score)
    if not products:
        return []

    query = " ".join([kw for kw, _ in sorted_keywords[:5]])
    q_emb = embedding_model.encode(query, convert_to_tensor=True)

    ranked = sorted(
        [(p, util.cos_sim(q_emb, p["embedding"]).item()) for p in products],
        key=lambda x: x[1], reverse=True
    )

    csv_path = category_to_file.get(allowed_category)
    df = pd.read_csv(csv_path)

    results = []
    for p, sim in ranked[:5]:
        name = p["name"]
        match = df[df["상품명"].str.contains(re.escape(name), na=False)]
        if not match.empty:
            row = match.iloc[0]
            results.append({
                "id": str(uuid.uuid4()),
                "name": name,
                "category": allowed_category,
                "imageUrl": row["이미지URL"],
                "price": row["가격"],
                "description": row["상품URL"]
            })
        else:
            results.append({
                "id": str(uuid.uuid4()),
                "name": name,
                "category": allowed_category,
                "imageUrl": None,
                "price": "정보 없음",
                "description": None
            })
    return results

if __name__ == "__main__":
    file_path = "chat_exam.txt"
    data_by_date = extract_kakao_dialogues(file_path)

    for date, msgs in sorted(data_by_date.items()):
        msgs = [m for m in msgs if is_valid_conversation(m)]
        if not msgs:
            continue

        full_text = " ".join(msgs)
        subject, main_cat = classify_topic(full_text)
        intimacy = classify_avg_score_from_pairs(msgs)
        keywords = extract_interest_weighted_keywords(msgs)

        print(f"\n📅 {date}  주제:{subject}  대분류:{main_cat}  친밀도:{intimacy}")
        for kw, sc in keywords[:5]:
            print(f" - {kw}: {sc:.2f}")

        print("\n🎁 추천 TOP 5:")
        recs = recommend_products_from_keywords(keywords, main_cat, intimacy)
        for p in recs:
            print(f" - {p['name']} | 가격: {p['price']} | 링크: {p['description']}")
