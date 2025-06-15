from flask import Flask, request, jsonify
import os
import uuid
import json
from flask_cors import CORS
import uuid


from final_test import extract_kakao_dialogues, is_valid_conversation, classify_topic, classify_avg_score_from_pairs, extract_interest_weighted_keywords, recommend_products_from_keywords

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploaded"
ANALYSIS_FOLDER = "analysis"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

def api_response(success, data=None, message=None, error=None):
    return jsonify({
        "success": success,
        "data": data,
        "message": message,
        "error": error
    })

@app.route("/api/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return api_response(False, error="파일이 없습니다.")

    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_FOLDER, file_id + ".txt")
    file.save(path)
    return api_response(True, data={"fileId": file_id}, message="파일 업로드 성공")

@app.route("/api/analyze", methods=["POST"])
def analyze_file():
    data = request.get_json()
    file_id = data.get("fileId")
    file_path = os.path.join(UPLOAD_FOLDER, file_id + ".txt")
    if not os.path.exists(file_path):
        return api_response(False, error="해당 파일을 찾을 수 없습니다.")

    data_by_date = extract_kakao_dialogues(file_path)
    result = []

    for date, msgs in sorted(data_by_date.items()):
        msgs = [m for m in msgs if is_valid_conversation(m)]
        if not msgs:
            continue
        full_text = " ".join(msgs)
        subject, main_cat = classify_topic(full_text)
        intimacy = classify_avg_score_from_pairs(msgs)
        keywords = extract_interest_weighted_keywords(msgs)

        result.append({
            "date": date,
            "subject": subject,
            "category": main_cat,
            "intimacy": intimacy,
            "keywords": [{"name": kw, "score": round(score, 2)} for kw, score in keywords[:5]]
        })

    return api_response(True, data=result, message="분석 완료")

@app.route("/api/recommendations", methods=["POST"])
def recommend_file():
    data = request.get_json()
    file_id = data.get("fileId")
    file_path = os.path.join(UPLOAD_FOLDER, file_id + ".txt")
    if not os.path.exists(file_path):
        return api_response(False, error="해당 파일을 찾을 수 없습니다.")

    data_by_date = extract_kakao_dialogues(file_path)
    result = []

    for date, msgs in sorted(data_by_date.items()):
        msgs = [m for m in msgs if is_valid_conversation(m)]
        if not msgs:
            continue

        full_text = " ".join(msgs)
        subject, main_cat = classify_topic(full_text)
        intimacy = classify_avg_score_from_pairs(msgs)
        keywords = extract_interest_weighted_keywords(msgs)

        # 추천 결과 → 바로 products로 받기 (이제 dict 포함됨)
        recs = recommend_products_from_keywords(
            sorted_keywords=keywords,
            allowed_category=main_cat,
            intimacy_score=intimacy
        )

        result.append({
            "date": date,
            "recommendations": recs  # [{id, name, imageUrl, ...}, ...]
        })

    return api_response(True, data=result, message="추천 완료")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

