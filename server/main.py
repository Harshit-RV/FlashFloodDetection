from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import base64
import os
from google import genai
from google.genai import types
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app)
load_dotenv()
UPLOAD_FOLDER = 'files'


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MONGO_URL=os.getenv('MONGO_URI')

client = MongoClient(MONGO_URL)
db = client["flood_detection"]
collection = db["flood_detection"]


def predict(file_path):
    client = genai.Client(
        api_key = os.getenv('GEMINI_API_KEY')
    )

    files = [
        client.files.upload(file=file_path),
    ]

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="""Given any image, return whether the image seems to be captured during a flood and return your reasoning. Also rate how severe the flood is based on the destruction nearby, the water level and other factors.
0 being non-harmful normal flash floods
100 being crazy harsh floods that destroys infra

1. boolean for whether there is a flood or not
2. severity of the flood out of 100
3. estimated water level
4. basic 50 words description of the image, the destruction caused
5. description of the image so I know which image you just analysed

example response:
{
    "isFlood": true,
    "severity": 80,
    "waterLevel": "The water level is approximately halfway up the cars, reaching the windows in some cases.",
    "description": "The image depicts a severe urban flood with numerous vehicles submerged. Debris like bicycles and furniture are scattered in the water. People are seen wading through the floodwaters or observing from higher ground, highlighting the disruption and danger caused.",
    "imageDescription": "The image shows a flooded urban area with many cars submerged."
}

always return JSON format data without any filler words as I'm directly going to run JSON.loads or JSON.parse on this response. THIS IS A STRICT INSTRUCTION AS THIS MISTAKE WILL BREAK THE FLOW OF THE CODE.
"""),
        ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text


@app.route('/upload', methods=['POST'])
def upload_image():
    try: 
        if 'image' not in request.files or 'metadata' not in request.files:
            return 'Request is missing image or metadata', 400
        image = request.files['image']

        if image.filename == '':
            print("No selected file")
            return 'No selected file', 400
    except Exception as e:
        print("error in body parsing: ", e)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
    except Exception as e:
        print("error in saving file: ", e)

    try: 
        responseFromGemini = predict(filepath)
        responseFromGemini = json.loads(responseFromGemini)
    except Exception as e:
        print("error in gemini response fetch and parse operations: ", e)


    try:
        metadata = request.files['metadata']
        metadata = json.loads(metadata.read())
    except Exception as e:
        print("error in metadata input parsing: ", e)

    try: 
        if collection.find_one({
            "id": metadata["id"]
        }):
            collection.update_one({
                "id": metadata["id"]
            }, {
                "$set": {
                    "lat": metadata["lat"],
                    "lng": metadata["long"],
                    "title": metadata["location"],
                    "status": "red" if responseFromGemini["isFlood"] else "green",
                    "image": filepath,
                    "description": responseFromGemini["description"],
                    "imageDescription": responseFromGemini["imageDescription"],
                    "severity": responseFromGemini["severity"],
                }
            }, upsert=True)
        else:
            collection.insert_one({
                "id": metadata["id"],
                "lat": metadata["lat"],
                "lng": metadata["long"],
                "title": metadata["location"],
                "status": "red" if responseFromGemini["isFlood"] else "green",
                "image": filepath,
                "description": responseFromGemini["description"],
                "imageDescription": responseFromGemini["imageDescription"],
                "severity": responseFromGemini["severity"],
            })
    except Exception as e:
        print("error in sending out response: ", e)

    return jsonify({"message": "Image uploaded successfully"}), 200

@app.route('/locations')
def locations():
    all_documents = list(collection.find({}))
    docs = []
    for document in all_documents:
        del document["_id"]
        print(document)
        docs.append(document)
    return jsonify(docs), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)