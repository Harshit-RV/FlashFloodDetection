from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime
import base64
import os
from google import genai
from google.genai import types
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import logging

# Configure logging for Render.com visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force Python to be unbuffered for immediate log visibility on Render
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


app = Flask(__name__)
CORS(app)
load_dotenv()

# Configure Flask logging to use the same logger
app.logger.handlers = logger.handlers
app.logger.setLevel(logging.INFO)

# Add startup message
logger.info("=== FLASK APPLICATION STARTING ===")
UPLOAD_FOLDER = '/tmp/files'

# Create upload directory
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload folder created/verified: {UPLOAD_FOLDER}")
except Exception as e:
    logger.error(f"Error creating upload folder: {e}")

# Environment variables with validation and detailed logging
MONGO_URL = os.getenv('MONGO_URI')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

logger.info("=== ENVIRONMENT CHECK ===")
logger.info(f"MONGO_URI set: {'Yes' if MONGO_URL else 'No'}")
logger.info(f"GEMINI_API_KEY set: {'Yes' if GEMINI_API_KEY else 'No'}")

if not MONGO_URL:
    logger.error("ERROR: MONGO_URI environment variable is not set!")
    raise ValueError("MONGO_URI environment variable is required")
if not GEMINI_API_KEY:
    logger.error("ERROR: GEMINI_API_KEY environment variable is not set!")
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Database connection with error handling
try:
    client = MongoClient(MONGO_URL)
    # Test the connection
    client.admin.command('ping')
    db = client["flood_detection"]
    collection = db["flood_detection"]
    logger.info("MongoDB connection successful!")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise


def predict(file_path):
    logger.info(f"Starting AI prediction for file: {file_path}")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY
        )
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        raise

    try:
        files = [
            client.files.upload(file=file_path),
        ]
        logger.info(f"File uploaded to Gemini successfully. URI: {files[0].uri}")
    except Exception as e:
        logger.error(f"Failed to upload file to Gemini: {e}")
        raise

    try:
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
        logger.info("Gemini content configuration prepared")
    except Exception as e:
        logger.error(f"Failed to prepare Gemini configuration: {e}")
        raise

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        logger.info(f"Gemini response received. Length: {len(response.text) if response.text else 0}")
        return response.text
    except Exception as e:
        logger.error(f"Failed to generate content with Gemini: {e}")
        raise


@app.route('/upload', methods=['POST'])
def upload_image():
    logger.info("=== UPLOAD REQUEST STARTED ===")
    try: 
        logger.info("Checking request files...")
        if 'image' not in request.files or 'metadata' not in request.files:
            logger.warning("Missing files in request")
            return jsonify({"error": "Request is missing image or metadata"}), 400
        image = request.files['image']
        logger.info(f"Image file received: {image.filename}")

        if image.filename == '':
            logger.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400
    except Exception as e:
        logger.error(f"Error in request parsing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to parse request body"}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image.save(filepath)
        logger.info(f"File saved to: {filepath}")
    except Exception as e:
        logger.error(f"Error in saving file: {e}")
        return jsonify({"error": "Failed to save uploaded file"}), 500

    try: 
        logger.info("Starting AI processing...")
        responseFromGemini = predict(filepath)
        logger.info(f"Raw Gemini response: {responseFromGemini[:200]}...")
        responseFromGemini = json.loads(responseFromGemini)
        logger.info("Gemini response received and parsed successfully")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in Gemini response: {e}")
        logger.error(f"Raw response was: {responseFromGemini if 'responseFromGemini' in locals() else 'No response received'}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to parse AI response"}), 500
    except Exception as e:
        logger.error(f"Error in Gemini processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to process image with AI"}), 500

    try:
        logger.info("Parsing metadata...")
        metadata = request.files['metadata']
        metadata_content = metadata.read()
        logger.info(f"Metadata content: {metadata_content}")
        metadata = json.loads(metadata_content)
        logger.info(f"Metadata parsed successfully: {metadata}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in metadata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Invalid JSON in metadata"}), 400
    except Exception as e:
        logger.error(f"Error in metadata parsing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to parse metadata"}), 400

    try: 
        logger.info("Saving to database...")
        # Store relative path instead of absolute path for better portability
        relative_filepath = f"files/{filename}"
        
        if collection.find_one({"id": metadata["id"]}):
            logger.info(f"Updating existing document for ID: {metadata['id']}")
            collection.update_one({
                "id": metadata["id"]
            }, {
                "$set": {
                    "lat": metadata["lat"],
                    "lng": metadata["long"],
                    "title": metadata["location"],
                    "status": "red" if responseFromGemini["isFlood"] else "green",
                    "image": relative_filepath,
                    "description": responseFromGemini["description"],
                    "imageDescription": responseFromGemini["imageDescription"],
                    "severity": responseFromGemini["severity"],
                    "timestamp": timestamp
                }
            }, upsert=True)
            logger.info("Document updated in database")
        else:
            logger.info(f"Inserting new document for ID: {metadata['id']}")
            collection.insert_one({
                "id": metadata["id"],
                "lat": metadata["lat"],
                "lng": metadata["long"],
                "title": metadata["location"],
                "status": "red" if responseFromGemini["isFlood"] else "green",
                "image": relative_filepath,
                "description": responseFromGemini["description"],
                "imageDescription": responseFromGemini["imageDescription"],
                "severity": responseFromGemini["severity"],
                "timestamp": timestamp
            })
            logger.info("New document inserted in database")
    except Exception as e:
        logger.error(f"Error in database operations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to save data to database"}), 500

    # Clean up temporary file after processing (optional for /tmp)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Temporary file {filepath} cleaned up")
    except Exception as e:
        logger.warning(f"Could not clean up temporary file: {e}")

    logger.info("=== UPLOAD REQUEST COMPLETED SUCCESSFULLY ===")
    return jsonify({"message": "Image uploaded successfully", "timestamp": timestamp}), 200

@app.route('/locations')
def locations():
    try:
        all_documents = list(collection.find({}))
        docs = []
        for document in all_documents:
            del document["_id"]
            logger.debug(f"Location document: {document}")
            docs.append(document)
        logger.info(f"Retrieved {len(docs)} location documents")
        return jsonify(docs), 200
    except Exception as e:
        logger.error(f"Error in fetching locations: {e}")
        return jsonify({"error": "Failed to fetch locations"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Server is running"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    logger.info(f"Starting Flask server on port {port}")
    logger.info("All systems initialized successfully!")
    app.run(host='0.0.0.0', port=port, debug=False)



