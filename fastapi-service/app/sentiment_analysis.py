import openai
import json
from app.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def analyze_text_sentiment(text: str):
    """OpenAI API를 사용하여 텍스트 감정 분석 및 JSON 형식으로 노래 추천"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 텍스트의 감정을 분석하고, 그 감정에 가장 잘 어울리는 최신 한국 노래 10곡을 JSON 형식으로 추천하는 AI야."},
                {"role": "user", "content": f"""
                다음 문장의 감정을 분석하고, 해당 감정에 맞는 최신 한국 노래 10곡을 추천해줘.
                노래는 발라드, 인디, R&B 등 다양한 장르에서 선택해야 해.
                JSON으로 반환해야 하고, 반드시 아래 형식을 따라야 해:
                
                ```json
                {{
                    "songs": [
                        {{"title": "노래 제목", "artist": "아티스트"}},
                        {{"title": "노래 제목", "artist": "아티스트"}}
                    ]
                }}
                ```
                
                JSON 외의 다른 문장은 포함하지 마.
                분석할 문장: "{text}"
                """}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        return {"error": str(e)}


def analyze_image_sentiment(image_data: bytes) -> dict:
    """OpenAI API를 사용하여 이미지 감정 분석 및 JSON 형식으로 노래 추천"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "너는 이미지의 감정을 분석하고 그 감정에 가장 잘 어울리는 최신 한국 노래 10곡을 JSON 형식으로 추천하는 AI야."},
                {"role": "user", "content": f"""
                다음 이미지의 감정을 분석하고, 해당 감정에 맞는 최신 한국 노래 10곡을 추천해줘.
                노래는 발라드, 인디, R&B 등 다양한 장르에서 선택해야 해.
                JSON으로 반환해야 하고, 반드시 아래 형식을 따라야 해:
                
                ```json
                {{
                    "songs": [
                        {{"title": "노래 제목", "artist": "아티스트"}},
                        {{"title": "노래 제목", "artist": "아티스트"}}
                    ]
                }}
                ```
                
                JSON 외의 다른 문장은 포함하지 마.
                """},
                {"role": "user", "content": image_data}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        return {"error": str(e)}