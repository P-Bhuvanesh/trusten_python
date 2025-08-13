"""
Without camera check on backend /status endpoint
Replaced cv2 with PIL
Added mail notification functionality
Admin priviledge added

"""


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import base64
import numpy as np
from PIL import Image
import io
import mediapipe as mp
from keras_facenet import FaceNet
import datetime
from scipy.spatial.distance import cosine
import traceback
import logging
from pymongo import MongoClient, errors as mongo_errors
from pymongo.errors import PyMongoError
import uvicorn 
import random
from dotenv import load_dotenv
from bson.json_util import dumps
import json
from email_utils import send_email_async
from fastapi import BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
import requests

load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='attendance_app.log'
)
logger = logging.getLogger(__name__)

class MediaPipeFilter(logging.Filter):
    def filter(self, record):
        return "inference_feedback_manager.cc" not in record.getMessage()

logger.addFilter(MediaPipeFilter())

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()
embedder = FaceNet() 

def get_database_connection():
    try:
        MONGO_URL = os.getenv("MONGO_URL")
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        
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

db_connection = get_database_connection()


def generate_unique_id():
    while True:
        user_id = random.randint(10000, 99999)
        existing_user = db_connection["users_collection"].find_one({"_id": user_id})
        if not existing_user:
            return user_id
        
class ImageData(BaseModel):
    image: str  

class User(BaseModel):
    name: str
    email: str
    designation: str
    department: str
    images: list[str] 

class LoginData(BaseModel):
    username: str
    password: str

def extract_embedding(image_data):
    """Extract facial embeddings from an image"""
    try:
        # Decode base64 image
        image_content = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_content)
        
        # Open with PIL
        pil_img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB (in case it's not)
        pil_img = pil_img.convert('RGB')
        
        # Convert to numpy array for MediaPipe
        img_array = np.array(pil_img)
        
        # Process with MediaPipe
        detections = face_detection.process(img_array)
        
        if detections.detections:
            for detection in detections.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img_array.shape 
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                
                # Extract face region from numpy array
                face = img_array[y:y+height, x:x+width]
                
                # Resize using PIL
                face_pil = Image.fromarray(face)
                face_resized = face_pil.resize((160, 160))
                
                # Convert back to numpy for FaceNet
                face_np = np.array(face_resized)
                face_recog = np.expand_dims(face_np, axis=0)
                
                # Get embedding
                embedding = embedder.embeddings(face_recog)[0]
                return embedding.tolist()
        return None
    except Exception as e:
        logger.error(f"Error in extract_embedding: {e}")
        traceback.print_exc()
        return None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trusten.vercel.app",
        "http://localhost:5173",
        "https://trusten-p-bhuvaneshs-projects.vercel.app",
        "https://trusten-git-main-p-bhuvaneshs-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"python server":"running"}

@app.post("/admin/login")
async def admin_login(data: LoginData):
    print(data)
    print(os.getenv("USERNAME"))
    print(data.username == os.getenv("USERNAME"))
    print(os.getenv("PASSWORD"))
    print(data.password == os.getenv("PASSWORD"))
    if data.username == os.getenv("ADMINUSER") and data.password == os.getenv("ADMINPASSWORD"):
        return {"success": True}
    return {"success": False}

@app.get("/status")
def get_status():
    if db_connection is None:
        database_status = False
    else:
        database_status = True
    return {
        "database": database_status,
        "server": True 
    }

def ping_server():
    try:
        res = requests.get("https://trusten-python.onrender.com/")
        print(f"Ping status: {res.status_code}")
    except Exception as e:
        print(f"Ping failed: {e}")

@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(ping_server, "interval", minutes=14)
    scheduler.start()

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

@app.post("/add_user")
async def add_user(user: User, background_tasks: BackgroundTasks):
    try:
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Check if user already exists
        existing_user = db_connection['users_collection'].find_one({
            "email": user.email
        })
        if existing_user:
            logger.warning(f"Duplicate user detected: {user.email}")
            raise HTTPException(status_code=400, detail="User already exists")

        embeddings_list = []
        for idx, img_data in enumerate(user.images[:10]):
            embedding = extract_embedding(img_data)
            if embedding:
                embeddings_list.append(embedding)

        if not embeddings_list:
            raise HTTPException(status_code=400, detail="No valid face detected in images")

        try:
            user_id = generate_unique_id()
            db_connection['users_collection'].insert_one({
                "user_id": user_id,
                "name": user.name,
                "email": user.email,
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

        # Send welcome email
        subject = f"Welcome to Trusten.Vision - You're Now Set to Check In/Out"
        body = f"""\
            Hello {user.name},

            Welcome aboard! You have been successfully added to the Trusten.Vision attendance system.

            Your unique ID is: **{user_id}**

            Your details:
            - Name: {user.name}
            - Designation :{user.designation}
            - Department :{user.department}

            You can now start using the application to check in and check out. Your attendance will be securely recorded and accessible by the admin team.

            If you have any questions or need assistance, feel free to reach out to our support team.

            Best regards,  
            Team Trusten.Vision
        """


        background_tasks.add_task(send_email_async, user.email, subject, body)

        return {
            "message": f"{user.name}, your ID is: {user_id}"
        }

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error in add_user: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")


@app.post("/check-in")
async def check_in(data: ImageData, background_tasks: BackgroundTasks):
    try:
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        embedding = extract_embedding(data.image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        users = db_connection['users_collection'].find({}, {"user_id": 1, "name": 1, "email": 1, "embeddings": 1})
        recognized_user = None
        user_details = None
        best_score = 1.0
        threshold = 0.7

        for user in users:
            stored_embeddings = user["embeddings"]
            if not stored_embeddings:
                continue

            distances = [cosine(embedding, e) for e in stored_embeddings]
            avg_distance = sum(distances) / len(distances)

            if avg_distance < threshold and avg_distance < best_score:
                best_score = avg_distance
                recognized_user = user["name"]
                user_details = user

        if recognized_user is None:
            raise HTTPException(status_code=400, detail="User not recognized")

        today = datetime.date.today().strftime("%Y-%m-%d")
        parsed_date = datetime.datetime.strptime(today, "%Y-%m-%d")
        reversed_format = parsed_date.strftime("%d %B %Y")

        now_time = datetime.datetime.now().strftime("%H:%M:%S")

        existing_record = db_connection['attendance_collection'].find_one({
            "user_id": user_details["user_id"],
            "date": today
        })

        if existing_record and existing_record.get("check_in"):
            logger.warning(f"Duplicate check-in attempt for {recognized_user}")
            raise HTTPException(status_code=400, detail=f"{recognized_user} has already checked in today.")

        db_connection['attendance_collection'].insert_one({
            "user_id": user_details["user_id"],
            "date": today,
            "check_in": now_time,
            "check_out": None
        })

        updated_embeddings = user_details["embeddings"]
        updated_embeddings.append(embedding)
        if len(updated_embeddings) > 10:  # rolling buffer size = 10
            updated_embeddings = updated_embeddings[-10:]

        db_connection['users_collection'].update_one(
            {"user_id": user_details["user_id"]},
            {"$set": {"embeddings": updated_embeddings}}
        )

        logger.info(f"{recognized_user} checked in at {now_time}")

        subject = f"Check-in Confirmation"

        body = f"""
            Hello {recognized_user},

            This is to confirm that you have successfully checked in on {reversed_format} at {now_time}.

            Thank you for using the Trusten.Vision Attendance System. If you did not perform this action, please contact your administrator immediately.

            Best regards,  
            Team Trusten devs
        """
        background_tasks.add_task(send_email_async, user_details["email"], subject, body)

        return {
            "message": f"{recognized_user} checked in at {now_time}"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in check_in: {e}")
        traceback.print_exc()
        raise

@app.post("/check-out")
async def check_out(data: ImageData, background_tasks: BackgroundTasks):
    try:
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        embedding = extract_embedding(data.image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        users = db_connection['users_collection'].find({}, {"user_id":1,"name": 1, "embeddings": 1, "email":1})
        recognized_user = None
        user_details = None
        min_distance = 0.45

        for user in users:
            for stored_embedding in user["embeddings"]:
                distance = cosine(embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_user = user["name"]
                    user_details = user

        if recognized_user is None:
            raise HTTPException(status_code=400, detail="User not recognized")

        today = datetime.date.today().strftime("%Y-%m-%d")
        parsed_date = datetime.datetime.strptime(today, "%Y-%m-%d")
        reversed_format = parsed_date.strftime("%d %B %Y")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")

        existing_record = db_connection['attendance_collection'].find_one({
            "user_id":user_details["user_id"],
            "date": today,
            "check_in": {"$exists": True} 
        })

        if not existing_record:
            message = f"{recognized_user} has not checked in today. \nYou must check in before checking out"
            raise HTTPException(status_code=400, detail=message)

        if existing_record.get("check_out"):
            message = f"{recognized_user} has already checked out today."
            raise HTTPException(status_code=400, detail=message)

        db_connection['attendance_collection'].update_one(
            {"_id": existing_record["_id"]},
            {"$set": {"check_out": now_time}}
        )

        updated_embeddings = user_details["embeddings"]
        updated_embeddings.append(embedding)
        if len(updated_embeddings) > 10:  # rolling buffer size = 10
            updated_embeddings = updated_embeddings[-10:]

        db_connection['users_collection'].update_one(
            {"user_id": user_details["user_id"]},
            {"$set": {"embeddings": updated_embeddings}}
        )

        subject = f"Check-out Confirmation"

        body = f"""
            Hello {recognized_user},

            This is to confirm that you have successfully checked out on {reversed_format} at {now_time}.

            Thank you for using the Trusten.Vision Attendance System. If you did not perform this action, please contact your administrator immediately.

            Best regards,  
            Team Trusten devs
        """
        background_tasks.add_task(send_email_async, user_details["email"], subject, body)

        success_message = f"Thank you, {recognized_user}. You have been checked out."

        logger.info(f"{recognized_user} checked out at {now_time}")
        return {"message": f"{recognized_user} checked out at {now_time}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in check_out: {e}")
        traceback.print_exc()
        raise

@app.get("/active_users")
async def get_active_users(department: str = Query(None), designation: str = Query(None)):
    try:
        # Ensure users_collection exists
        if db_connection is None:
            raise HTTPException(status_code=500, detail="Database connection error: users_collection is None")

        query = {}
        if department:
            query["department"] = department
        if designation:
            query["designation"] = designation

        users_cursor = db_connection["users_collection"].find(query, {"_id": 0,"user_id":1, "name": 1,"email":1, "department": 1, "designation": 1})
        users = list(users_cursor)

        if not users:
            return {"message": "No active users"}

        return {"active_users": users}

    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/get_users_check")
def get_users():
    users = list(db_connection["users_collection"].find({}, {"_id": 0, "user_id": 1, "name": 1, "email": 1}))
    return {"users": json.loads(dumps(users))}

@app.get("/get_users")
def get_users_id():
    users = list(db_connection["users_collection"].find({}, {"_id": 0, "user_id": 1, "name": 1}))
    return json.loads(dumps(users))

@app.get("/get-attendance/{user_id}")
def get_attendance(user_id: int):
    attendance_records = list(db_connection["attendance_collection"].find({"user_id": user_id}, {"_id": 0, "date": 1, "check_in": 1, "check_out": 1}))
    return json.loads(dumps(attendance_records))


if __name__ == "__main__":
    print("\n\n!!!!!!!!!! PYTHON SERVER IS UP !!!!!!!!!!!!\n\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
