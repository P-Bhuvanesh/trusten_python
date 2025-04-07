"""
Has pytts, cv2. 

No /status check enpoint -> old ones

"""


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import base64
import numpy as np
import cv2
import mediapipe as mp
from keras_facenet import FaceNet
import datetime
from scipy.spatial.distance import cosine
import traceback
import logging
from pymongo import MongoClient, errors as mongo_errors
from pymongo.errors import PyMongoError
import pyttsx3
import uvicorn 
import random
from dotenv import load_dotenv


load_dotenv()

PORT = int(os.getenv("PORT"))
URL = os.getenv("MONGO_URL")



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='attendance_app.log'
)
logger = logging.getLogger(__name__)

# Filter out MediaPipe warnings
class MediaPipeFilter(logging.Filter):
    def filter(self, record):
        return "inference_feedback_manager.cc" not in record.getMessage()

logger.addFilter(MediaPipeFilter())


app = FastAPI()

# Improved CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status":"healthy"}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tts_engine = pyttsx3.init()
voices = tts_engine.getProperty("voices")

# Set a different voice (try different indexes)
tts_engine.setProperty("voice", voices[1].id)
tts_engine.setProperty('rate', 200)  # Speed of speech
tts_engine.setProperty('volume', 1.0)  # Volume level

def speak_text(text):
    """Convert text to speech"""
    tts_engine.say(text)
    tts_engine.runAndWait()


# MongoDB Setup with improved connection handling
def get_database_connection():
    try:
        MONGO_URI = os.getenv("MONGO_URL")
        print(f"\n\n\n\n{MONGO_URI}\n\n\n\n")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.admin.command('ismaster')
        
        db = client["userlogs"]
        return {
            "client": client,
            "users_collection": db["users"],
            "attendance_collection": db["attendance"]
        }
    except mongo_errors.ConnectionFailure:
        logger.error("Failed to connect to MongoDB")
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to MongoDB: {e}")
        return None

# Global variable to store database connection
db_connection = get_database_connection()


def generate_unique_id():
    while True:
        user_id = random.randint(10000, 99999)
        existing_user = db_connection["users_collection"].find_one({"_id": user_id})
        if not existing_user:
            return user_id
        
class ImageData(BaseModel):
    image: str  # Base64 encoded image

class User(BaseModel):
    name: str
    designation: str
    department: str
    images: list[str]  # Base64 encoded images

# # Initialize Mediapipe and FaceNet
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()
embedder = FaceNet()

def extract_embedding(image_data):
    """Extract facial embeddings from an image"""
    try:
        nparr = np.frombuffer(base64.b64decode(image_data.split(",")[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Add error handling for image decoding
        if img is None:
            logger.error("Failed to decode image")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = face_detection.process(img)
        
        if detections.detections:
            for detection in detections.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape 
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin*h), int(bboxC.width * w), int(bboxC.height * h)
                face = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (160, 160))
                face_recog = np.expand_dims(face_resized, axis=0)
                embedding = embedder.embeddings(face_recog)[0]
                return embedding.tolist()
        return None
    except Exception as e:
        logger.error(f"Error in extract_embedding: {e}")
        traceback.print_exc()
        return None

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to log and return detailed error messages"""
    logger.error(f"Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500, 
        content={
            "message": "An unexpected error occurred",
            "error": str(exc),
            "details": traceback.format_exc()
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server and database connection"""
    if db_connection is None:
        return JSONResponse(
            status_code=500, 
            content={"status": "Database connection failed"}
        )
    return {"status": "healthy"}

@app.post("/add_user")
async def add_user(user: User):
    try:
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Check if user already exists
        existing_user = db_connection['users_collection'].find_one({
            "name": user.name,
            "designation": user.designation,
            "department": user.department
        })
        if existing_user:
            logger.warning(f"Duplicate user detected: {user.name}, {user.designation}, {user.department}")
            raise HTTPException(status_code=400, detail="User already exists")

        # user_id = str(uuid.uuid4())
        # folder_path = f"user_images/{user.name.replace(' ', '_')}"
        # os.makedirs(folder_path, exist_ok=True)

        embeddings_list = []
        for idx, img_data in enumerate(user.images[:10]):  # Limit to 10 images
            # img_path = os.path.join(folder_path, f"image_{idx + 1}.jpg")
            # try:
            #     with open(img_path, "wb") as img_file:
            #         img_file.write(base64.b64decode(img_data.split(",")[1]))
            # except Exception as file_error:
            #     logger.error(f"Error saving image for {user.name}: {file_error}")
            #     continue

            embedding = extract_embedding(img_data)
            if embedding:
                embeddings_list.append(embedding)

        if not embeddings_list:
            raise HTTPException(status_code=400, detail="No valid face detected in images")
        try:
            user_id = generate_unique_id()
            db_connection['users_collection'].insert_one({
                "_id": user_id,
                "name": user.name,
                "designation": user.designation,
                "department": user.department,
                "embeddings": embeddings_list
            })
        except mongo_errors.DuplicateKeyError:
            logger.error(f"Duplicate key error while inserting user {user.name}")
            raise HTTPException(status_code=400, detail="User insertion failed due to duplicate ID")
        except Exception as db_error:
            logger.error(f"Database error while inserting user {user.name}: {db_error}")
            raise HTTPException(status_code=500, detail="Database error while saving user")

        return {"message": f"User {user.name} added successfully with facial embeddings!"}

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error in add_user: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.post("/recognize")
async def recognize_face(data: ImageData):
    try:
        # Check database connection
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        embedding = extract_embedding(data.image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        users = db_connection['users_collection'].find({}, {"name": 1, "embeddings": 1})

        recognized_user = None
        min_distance = 0.6  # Threshold for recognition

        for user in users:
            for stored_embedding in user["embeddings"]:
                distance = cosine(embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_user = user["name"]

        if recognized_user:
            return {"name": recognized_user}
        else:
            return {"name": "Unknown"}
    except Exception as e:
        logger.error(f"Error in recognize_face: {e}")
        traceback.print_exc()
        raise
    

@app.post("/check-in")
async def check_in(data: ImageData):
    try:
        # Check database connection
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        embedding = extract_embedding(data.image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        # user = await recognize_face(data)
        # name = user["name"]

        users = db_connection['users_collection'].find({}, {"name": 1, "designation": 1, "department": 1, "embeddings": 1})
        recognized_user = None
        user_details = None
        min_distance = 0.6
        for user in users:
            for stored_embedding in user["embeddings"]:
                distance = cosine(embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_user = user["name"]
                    user_details = user
        
        if recognized_user == None:
            speak_text("User not recognized. Please add user and try again.")
            raise HTTPException(status_code=400, detail="User not recognized")

        today = datetime.date.today().strftime("%Y-%m-%d")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")

        existing_record = db_connection['attendance_collection'].find_one({"name": recognized_user,"designation":user_details["designation"],"department":user_details["department"], "date": today})

        if existing_record:
            # Log the duplicate check-in attempt
            logger.warning(f"Duplicate check-in attempt for {recognized_user}")
            message = f"{recognized_user} has already checked in today."
            speak_text(message)
            raise HTTPException(status_code=400, detail=f"{recognized_user} has already checked in today")

        db_connection['attendance_collection'].insert_one({
            "_id":user_details["_id"],
            "name": recognized_user,
            "date": today,
            "designation": user_details["designation"],
            "department": user_details["department"],
            "check_in": now_time,
            "check_out": None
        })

        success_message = f"Hello, {recognized_user}. Good morning! You have been checked in."
        speak_text(success_message)

        logger.info(f"{recognized_user} checked in at {now_time}")
        return {"message": f"{recognized_user} checked in at {now_time}"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in check_in: {e}")
        traceback.print_exc()
        raise

@app.post("/check-out")
async def check_out(data: ImageData):
    try:
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        embedding = extract_embedding(data.image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        users = db_connection['users_collection'].find({}, {"name": 1, "designation": 1, "department": 1, "embeddings": 1})
        recognized_user = None
        user_details = None
        min_distance = 0.95

        for user in users:
            for stored_embedding in user["embeddings"]:
                distance = cosine(embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_user = user["name"]
                    user_details = user

        if recognized_user is None:
            speak_text("User not recognized. Please try again after adding user.")
            raise HTTPException(status_code=400, detail="User not recognized")

        today = datetime.date.today().strftime("%Y-%m-%d")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")

        existing_record = db_connection['attendance_collection'].find_one({
            "name": recognized_user,
            "designation": user_details["designation"],
            "department": user_details["department"],
            "date": today,
            "check_in": {"$exists": True} 
        })

        if not existing_record:
            message = f"{recognized_user} has not checked in today."
            speak_text(message)
            raise HTTPException(status_code=400, detail=f"{recognized_user} has not checked in today")

        if existing_record.get("check_out"):
            message = f"{recognized_user} has already checked out today."
            speak_text(message)
            raise HTTPException(status_code=400, detail=message)

        db_connection['attendance_collection'].update_one(
            {"_id": existing_record["_id"]}, 
            {"$set": {"check_out": now_time}}
        )

        success_message = f"Thank you, {recognized_user}. You have been checked out."
        speak_text(success_message)

        logger.info(f"{recognized_user} checked out at {now_time}")
        return {"message": f"{recognized_user} checked out at {now_time}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in check_out: {e}")
        traceback.print_exc()
        raise

db_connection = get_database_connection()
if db_connection is None or "users_collection" not in db_connection:
    raise Exception("Failed to connect to MongoDB")

users_collection = db_connection.get("users_collection")

@app.get("/active_users")
async def get_active_users(department: str = Query(None), designation: str = Query(None)):
    try:
        # Ensure users_collection exists
        if users_collection is None:
            raise HTTPException(status_code=500, detail="Database connection error: users_collection is None")

        query = {}
        if department:
            query["department"] = department
        if designation:
            query["designation"] = designation

        users_cursor = users_collection.find(query, {"_id": 0, "name": 1, "department": 1, "designation": 1})
        users = list(users_cursor)

        if not users:
            return {"message": "No active users"}

        return {"active_users": users}

    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("!!!!!!!!!! PYTHON SERVER IS UP !!!!!!!!!!!!")
    print("🔥 FastAPI server is starting... Visit http://localhost:8000/docs to check.")
    uvicorn.run(app, host="127.0.0.1", port=PORT)