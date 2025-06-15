import re
from collections import defaultdict
from keybert import KeyBERT
from konlpy.tag import Okt
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel

# 관심 분류 모델 로드
model_name = "skt/kobert-base-v1"
tokenizer = KoBERTTokenizer.from_pretrained(model_name)
interest_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
interest_model.load_state_dict(torch.load("./kobert_importance.pth", map_location="cpu"))
interest_model.eval()

# 키워드 추출 및 임베딩 모델
kw_model = KeyBERT(model="distiluse-base-multilingual-cased-v1")
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
okt = Okt()

# 주제 분류 모델 정의
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

# 주제 id → 이름, 대분류 매핑
subject_id2name = {
    0: "미용", 1: "스포츠/레저", 2: "교육", 3: "가족", 5: "영화/만화",
    6: "교통", 7: "여행", 8: "회사/아르바이트", 9: "건강", 10: "연애/결혼",
    11: "게임", 12: "계절/날씨", 13: "방송/연예", 14: "사회이슈",
    15: "주거와 생활", 16: "반려동물", 17: "군대", 18: "식음료"
}

subject_to_main_category = {
    0: "뷰티", 1: "레저/스포츠", 2: "리빙/도서", 3: "디지털/가전", 5: "패션",
    6: "디지털/가전", 7: "레저/스포츠", 8: "리빙/도서", 9: "식품", 10: "패션",
    11: "디지털/가전", 12: "식품", 13: "패션", 14: "리빙/도서",
    15: "리빙/도서", 16: "유아동/반려", 17: "패션", 18: "패션"
}

# 분류 모델 로드
topic_tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)
topic_model = KoBertExtendedModel()
topic_model.load_state_dict(torch.load("kobert_extended_with_subject.pth", map_location="cpu"), strict=False)
topic_model.eval()

# 불용어 로드
with open("stopwords-ko.txt", encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())

def classify_interest(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = interest_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return int(torch.argmax(probs, dim=1))

def classify_topic(sentence):
    inputs = topic_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        _, _, subject_logits = topic_model(input_ids, attention_mask)
        subject_id = torch.argmax(subject_logits, dim=1).item()
        return subject_id2name.get(subject_id, "알 수 없음"), subject_to_main_category.get(subject_id, "없음")

def extract_kakao_dialogues(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    data_by_date = defaultdict(list)
    for line in lines:
        match = re.search(r"(\d{4})년 (\d{1,2})월 (\d{1,2})일", line)
        if match:
            y, m, d = match.groups()
            current_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        elif re.search(r"[오전|오후]+\s*\d{1,2}:\d{2},\s*[^:]+:", line):
            msg = re.sub(r"^\d{4}\. \d{1,2}\. \d{1,2}\. [오전|오후]+\s*\d{1,2}:\d{2},\s*[^:]+:\s*", "", line).strip()
            if len(msg) > 0:
                data_by_date[current_date].append(msg)
    return data_by_date

def is_valid_conversation(msg):
    return bool(re.search(r"[가-힣]", msg)) and not re.search(r"https?://|총\s*금액", msg)

def extract_interest_weighted_keywords(sentences):
    keyword_scores = defaultdict(float)
    for sentence in sentences:
        label = classify_interest(sentence)
        nouns = {n for n in okt.nouns(sentence) if n not in stopwords and len(n) > 1}
        keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=5)

        for kw, score in keywords:
            tokens = kw.split()
            if all(token in nouns for token in tokens):
                multiplier = 2.5 if len(tokens) > 1 else 2.0
                final_score = score * (multiplier if label == 1 else 0.5)
                keyword_scores[kw] += final_score

        for noun in nouns:
            add_score = 0.3 if label == 1 else 0.1
            keyword_scores[noun] += add_score

    filtered_keywords = [(kw, sc) for kw, sc in keyword_scores.items()
                      if all(not re.search(r"(다|어|지|음)$", token) for token in kw.split())]
    return sorted(filtered_keywords, key=lambda x: x[1], reverse=True)

# 대분류 → 대응 파일 경로 매핑
category_to_file = {
    "뷰티": "category_files/beauty.csv",
    "레저/스포츠": "category_files/sport.csv",
    "리빙/도서": "category_files/living.csv",
    "디지털/가전": "category_files/digital.csv",
    "패션": "category_files/fashion.csv",
    "유아동/반려": "category_files/baby.csv"
}

product_list = []
product_embeddings = []

product_embeddings = [
    {
        "name": p["name"],
        "embedding": embedding_model.encode(p["keywords"], convert_to_tensor=True),
        "category": p["category"]
    }
    for p in product_list
]

def recommend_products_from_keywords(sorted_keywords, allowed_category=None):
    global product_list, product_embeddings

    if allowed_category:
        csv_path = category_to_file.get(allowed_category)
        if not csv_path:
            print(f"[!] '{allowed_category}' 카테고리에 해당하는 CSV 파일이 없습니다. 건너뜁니다.")
            return []

        product_df = pd.read_csv(csv_path)
        product_list = [
            {
                "name": row["상품명"],
                "keywords": row["keywords"],
                "category": row["대분류"],
                "imageUrl": row["이미지URL"],
                "price": row["가격"],
                "description": f"{row['브랜드']} - {row['상품URL']}"
            }
            for _, row in product_df.iterrows()
            if pd.notna(row.get("상품명"))
            and pd.notna(row.get("keywords"))
            and pd.notna(row.get("이미지URL"))
            and pd.notna(row.get("가격"))
            and pd.notna(row.get("브랜드"))
            and pd.notna(row.get("상품URL"))
        ]

        product_embeddings = [
            {
                "name": p["name"],
                "embedding": embedding_model.encode(p["keywords"], convert_to_tensor=True),
                "category": p["category"],
                "imageUrl": p["imageUrl"],
                "price": p["price"],
                "description": p["description"]
            }
            for p in product_list
        ]

    query = " ".join([kw for kw, _ in sorted_keywords[:5]])
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    scores = []
    for prod in product_embeddings:
        if allowed_category and prod["category"] != allowed_category:
            continue
        score = util.cos_sim(query_embedding, prod["embedding"]).item()
        scores.append({
            "name": prod["name"],
            "score": score,
            "category": prod["category"],
            "imageUrl": prod["imageUrl"],
            "price": prod["price"],
            "description": prod["description"]
        })

    return sorted(scores, key=lambda x: x["score"], reverse=True)



# 파일 상단에 이미 존재하는 import와 모델 로딩 코드 생략

def analyze_kakao_file(file_path):
    data_by_date = extract_kakao_dialogues(file_path)
    results = []

    for date, messages in sorted(data_by_date.items()):
        filtered_msgs = [msg for msg in messages if is_valid_conversation(msg)]
        if len(filtered_msgs) == 0:
            continue

        full_text = " ".join(filtered_msgs)
        subject_name, main_category = classify_topic(full_text)
        keywords = extract_interest_weighted_keywords(filtered_msgs)

        results.append({
            "date": date,
            "subject": subject_name,
            "category": main_category,
            "keywords": keywords[:5]
        })
    return results

def recommend_from_analysis(analysis_result, filters=None):
    filters = filters or {}
    all_recommendations = []

    for entry in analysis_result:
        keywords = entry.get("keywords", [])
        category = entry.get("category", None)

        if not keywords or not category:
            continue

        recommendations = recommend_products_from_keywords(keywords, allowed_category=category)
        all_recommendations.append({
            "date": entry["date"],
            "category": category,
            "recommendations": recommendations[:5]  # 상위 5개만 리턴
        })

    return all_recommendations

