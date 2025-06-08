from flask import Flask, request, jsonify
from final import analyze_kakao_file, recommend_from_analysis
import os
import uuid
import json

app = Flask(__name__)
UPLOAD_FOLDER = "uploaded"
ANALYSIS_FOLDER = "analysis"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

# 공통 응답 포맷 함수
def api_response(success, data=None, message=None, error=None):
    return jsonify({
        "success": success,
        "data": data,
        "message": message,
        "error": error
    })


# 파일 업로드
@app.route("/api/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return api_response(False, error="파일이 없습니다.")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, file_id + ".txt")
    file.save(file_path)

    return api_response(True, message="업로드 성공", data={"fileId": file_id})


# 대화 분석
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    file_id = data.get("fileId")

    file_path = os.path.join(UPLOAD_FOLDER, file_id + ".txt")
    if not os.path.exists(file_path):
        return api_response(False, error="해당 파일이 존재하지 않습니다.")

    try:
        analysis_result = analyze_kakao_file(file_path)

        # 저장 (다음 요청을 위한 캐싱 용도)
        with open(os.path.join(ANALYSIS_FOLDER, file_id + ".json"), "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

        # 변환 (프론트 요구 형식에 맞게 단순화 예시)
        response_data = {
    "keywords": [
        {"name": kw, "value": round(score, 2)}
        for entry in analysis_result
        for kw, score in entry["keywords"]
    ],
    "relationship": [
        {
            "date": entry["date"],
            "intimacy": min(100, len(entry["keywords"]) * 10),
            "trend": min(100, len(entry["keywords"]) * 5)
        }
        for entry in analysis_result
    ]
}
        return api_response(True, data=response_data)

    except Exception as e:
        return api_response(False, error=str(e))


# 선물 추천
@app.route("/api/recommendations", methods=["POST"])
def recommend():
    data = request.get_json()
    file_id = data.get("fileId")
    page = int(data.get("page", 1))
    limit = int(data.get("limit", 5))
    category = data.get("category")
    max_price = data.get("maxPrice")

    try:
        # 분석 결과 로드
        analysis_path = os.path.join("analysis", file_id + ".json")
        if not os.path.exists(analysis_path):
            return api_response(False, error="분석 데이터가 없습니다.")

        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis_result = json.load(f)

        recommendations = recommend_from_analysis(analysis_result, filters={
            "category": category,
            "maxPrice": max_price
        })

        flat_list = []
        for rec in recommendations:
            for item in rec["recommendations"]:
                flat_list.append({
                    "id": str(uuid.uuid4()),
                    "name": item["name"],
                    "price": item["price"],
                    "imageUrl": item["imageUrl"],
                    "description": item["description"],
                    "category": item["category"],
                })

        paged = flat_list[(page - 1) * limit: page * limit]
        total_pages = (len(flat_list) + limit - 1) // limit

        return api_response(True, data={
            "gifts": paged,
            "totalCount": len(flat_list),
            "currentPage": page,
            "totalPages": total_pages
        })

    except Exception as e:
        return api_response(False, error=str(e))



if __name__ == "__main__":
    app.run(debug=True)
