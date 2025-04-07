"""
Any optimizations to be done? to make it available for deployment and hosting


Enhanced main.py by the claude. Added more validataion and logs

Error Handling and Logging:

Add structured logging with request IDs
Improve exception handling in database operations


Performance Optimizations:

Add caching for embedding comparisons
Optimize the face recognition threshold based on production data


Security Enhancements:

Add rate limiting to prevent brute force attacks
Sanitize inputs more strictly


Deployment Readiness:

Add health checks
Implement graceful shutdown
Configure CORS more restrictively
Fix the uvicorn.run parameters

"""


from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator
from typing import List, Dict, Optional, Any
import os
import base64
import numpy as np
from PIL import Image
import io
import mediapipe as mp
from keras_facenet import FaceNet
import datetime
import time
from scipy.spatial.distance import cosine
import traceback
import logging
import uuid
from pymongo import MongoClient, errors as mongo_errors
from pymongo.errors import PyMongoError
import uvicorn 
import random
from dotenv import load_dotenv
from bson.json_util import dumps
import json
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from functools import lru_cache

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
    filename='attendance_app.log'
)

# Add request_id to log records
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'no_request_id')
        return True

class MediaPipeFilter(logging.Filter):
    def filter(self, record):
        return "inference_feedback_manager.cc" not in record.getMessage()

logger = logging.getLogger(__name__)
logger.addFilter(RequestIdFilter())
logger.addFilter(MediaPipeFilter())

# Initialize FastAPI app
app = FastAPI(
    title="Attendance System API",
    description="Face recognition based attendance system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware for request ID
class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        # Use a context variable to store request_id
        request.state.request_id = request_id
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

# Add middleware
app.add_middleware(RequestIdMiddleware)

# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests=100, window_size=60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests = {}
        
    async def dispatch(self, request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        self.requests = {ip: [t for t in times if current_time - t < self.window_size] 
                        for ip, times in self.requests.items()}
        
        # Check if client exceeded rate limit
        if client_ip in self.requests:
            if len(self.requests[client_ip]) >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests, please try again later."}
                )
            self.requests[client_ip].append(current_time)
        else:
            self.requests[client_ip] = [current_time]
            
        return await call_next(request)

app.add_middleware(RateLimitMiddleware, max_requests=60, window_size=60)  # 60 requests per minute

# Set up CORS - restrict in production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add trusted host middleware for production security
if os.getenv("ENVIRONMENT", "development") == "production":
    trusted_hosts = os.getenv("TRUSTED_HOSTS", "localhost").split(",")
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=trusted_hosts
    )

# Setup TensorFlow environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Database connection with caching
@lru_cache(maxsize=1)
def get_database_connection():
    """Get database connection with retry logic and connection pooling"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            MONGO_URL = os.getenv("MONGO_URL")
            if not MONGO_URL:
                logger.error("MONGO_URL environment variable not set")
                return None
                
            client = MongoClient(
                MONGO_URL, 
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
                maxPoolSize=50,  # Connection pooling for better performance
                retryWrites=True
            )
            
            # Test connection
            client.admin.command('ismaster')
            
            db = client["userlogs"]
            return {
                "client": client,
                "users_collection": db["users"],
                "attendance_collection": db["attendance"]
            }
        except mongo_errors.ConnectionFailure as e:
            if attempt < max_retries - 1:
                logger.warning(f"MongoDB connection attempt {attempt+1} failed, retrying in {retry_delay}s: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed to connect to MongoDB after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return None

db_connection = get_database_connection()

# Generate user IDs
def generate_unique_id():
    """Generate unique user ID with collision detection"""
    if db_connection is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
        
    max_attempts = 10
    for _ in range(max_attempts):
        user_id = random.randint(10000, 99999)
        existing_user = db_connection["users_collection"].find_one({"user_id": user_id})
        if not existing_user:
            return user_id
            
    # If we reach here, we couldn't generate a unique ID after max_attempts
    logger.error("Failed to generate unique ID after multiple attempts")
    raise HTTPException(status_code=500, detail="Failed to generate unique ID")

# Pydantic models with validation
class ImageData(BaseModel):
    image: str
    
    @validator('image')
    def validate_image(cls, v):
        if not v.startswith('data:image/'):
            raise ValueError('Not a valid base64 image format')
        return v

class User(BaseModel):
    name: str
    email: EmailStr
    designation: str
    department: str
    images: List[str]
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters')
        return v
    
    @validator('images')
    def validate_images(cls, images):
        if not images or len(images) < 1:
            raise ValueError('At least one image is required')
        for img in images:
            if not img.startswith('data:image/'):
                raise ValueError('Not a valid base64 image format')
        return images

# Request ID dependency
async def get_request_id(request: Request):
    return request.state.request_id

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    db_status = db_connection is not None
    
    return {
        "status": "healthy" if db_status else "degraded",
        "timestamp": datetime.datetime.now().isoformat(),
        "database": db_status,
        "version": "1.0.0"
    }

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/status")
def get_status():
    """System component status check"""
    if db_connection is None:
        database_status = False
    else:
        database_status = True
    return {
        "database": database_status,
        "server": True,  # Since endpoint worked, server is running
        "timestamp": datetime.datetime.now().isoformat()
    }

# Initialize face detection models
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
embedder = FaceNet()

# LRU cache for face embeddings to improve performance
@lru_cache(maxsize=100)
def get_face_embedding_cached(image_hash):
    """Cached version of face embedding extraction to improve performance"""
    # This is a placeholder - we can't directly cache the function with image data
    # The actual implementation would need a different approach
    pass

# Extract face embeddings from image
def extract_embedding(image_data, request_id="unknown"):
    """Extract facial embeddings from an image"""
    try:
        # Decode base64 image
        try:
            image_content = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_content)
        except IndexError:
            logger.error(f"[{request_id}] Invalid image format: missing comma in base64 string")
            return None
        except base64.binascii.Error:
            logger.error(f"[{request_id}] Invalid base64 encoding in image")
            return None
        
        # Open with PIL
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
        except IOError:
            logger.error(f"[{request_id}] Failed to open image data")
            return None
        
        # Convert to RGB (in case it's not)
        pil_img = pil_img.convert('RGB')
        
        # Convert to numpy array for MediaPipe
        img_array = np.array(pil_img)
        
        # Process with MediaPipe
        detections = face_detection.process(img_array)
        
        if not detections.detections:
            logger.warning(f"[{request_id}] No face detected in image")
            return None
            
        # Get the detection with highest confidence if multiple faces detected
        detection = max(detections.detections, key=lambda d: d.score[0]) if len(detections.detections) > 1 else detections.detections[0]
        
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = img_array.shape 
        x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        if width <= 0 or height <= 0:
            logger.warning(f"[{request_id}] Invalid face bounding box")
            return None
        
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
    except Exception as e:
        logger.error(f"[{request_id}] Error in extract_embedding: {e}")
        traceback.print_exc()
        return None

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log and return detailed error messages"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"[{request_id}] Unhandled exception: {exc}")
    traceback.print_exc()
    
    # In production, don't return the detailed error trace
    if os.getenv("ENVIRONMENT", "development") == "production":
        return JSONResponse(
            status_code=500, 
            content={"message": "An unexpected error occurred", "request_id": request_id}
        )
    else:
        return JSONResponse(
            status_code=500, 
            content={
                "message": "An unexpected error occurred",
                "error": str(exc),
                "details": traceback.format_exc(),
                "request_id": request_id
            }
        )

# API endpoints
@app.post("/add_user")
async def add_user(user: User, request: Request):
    """Register a new user with face embeddings"""
    request_id = request.state.request_id
    try:
        if db_connection is None:
            raise HTTPException(status_code=503, detail="Database connection failed")

        # Check if user already exists
        existing_user = db_connection['users_collection'].find_one({
            "email": user.email
        })
        if existing_user:
            logger.warning(f"[{request_id}] Duplicate user detected: {user.email}")
            raise HTTPException(status_code=400, detail="User already exists")

        embeddings_list = []
        for idx, img_data in enumerate(user.images[:10]):  # Limit to 10 images
            embedding = extract_embedding(img_data, request_id)
            if embedding:
                embeddings_list.append(embedding)

        if not embeddings_list:
            raise HTTPException(status_code=400, detail="No valid face detected in images")
        
        try:
            user_id = generate_unique_id()
            # Create a document with index
            db_connection['users_collection'].insert_one({
                "user_id": user_id,
                "name": user.name,
                "email": user.email,
                "designation": user.designation,
                "department": user.department,
                "embeddings": embeddings_list,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            })
        except mongo_errors.DuplicateKeyError:
            logger.error(f"[{request_id}] Duplicate key error while inserting user {user.name}")
            raise HTTPException(status_code=400, detail="User insertion failed due to duplicate ID")
        except Exception as db_error:
            logger.error(f"[{request_id}] Database error while inserting user {user.name}: {db_error}")
            raise HTTPException(status_code=500, detail="Database error while saving user")

        logger.info(f"[{request_id}] User {user.name} (ID: {user_id}) successfully added")
        return {"message": f"{user.name} your ID is: {user_id}", "user_id": user_id}

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in add_user: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.post("/recognize")
async def recognize_face(data: ImageData, request: Request):
    """Recognize a face from the provided image"""
    request_id = request.state.request_id
    try:
        # Check database connection
        if db_connection is None:
            raise HTTPException(status_code=503, detail="Database connection failed")

        embedding = extract_embedding(data.image, request_id)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        users = db_connection['users_collection'].find({}, {"name": 1, "embeddings": 1})

        recognized_user = None
        recognized_user_id = None
        min_distance = 0.6  # Threshold for recognition - adjust based on testing

        for user in users:
            for stored_embedding in user["embeddings"]:
                distance = cosine(embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_user = user["name"]
                    recognized_user_id = user.get("user_id")

        if recognized_user:
            logger.info(f"[{request_id}] User {recognized_user} recognized with confidence {1-min_distance:.2f}")
            return {
                "name": recognized_user, 
                "confidence": 1-min_distance,
                "user_id": recognized_user_id
            }
        else:
            logger.info(f"[{request_id}] No user recognized (min distance: {min_distance:.2f})")
            return {"name": "Unknown", "confidence": 0}
    except Exception as e:
        logger.error(f"[{request_id}] Error in recognize_face: {e}")
        traceback.print_exc()
        raise

@app.post("/check-in")
async def check_in(data: ImageData, request: Request):
    """Register user check-in attendance"""
    request_id = request.state.request_id
    try:
        # Check database connection
        if db_connection is None:
            raise HTTPException(status_code=503, detail="Database connection failed")
        
        embedding = extract_embedding(data.image, request_id)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        users = db_connection['users_collection'].find({}, {"user_id":1, "name": 1, "embeddings": 1})
        recognized_user = None
        user_details = None
        min_distance = 0.45  # Lower threshold for increased security during check-in
        
        for user in users:
            for stored_embedding in user["embeddings"]:
                distance = cosine(embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_user = user["name"]
                    user_details = user
        
        if recognized_user is None:
            logger.warning(f"[{request_id}] Unrecognized user attempted check-in")
            raise HTTPException(status_code=400, detail="User not recognized")

        today = datetime.date.today().strftime("%Y-%m-%d")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")
        now_full = datetime.datetime.now()

        existing_record = db_connection['attendance_collection'].find_one({
            "user_id": user_details["user_id"], 
            "date": today
        })

        if existing_record:
            logger.warning(f"[{request_id}] Duplicate check-in attempt for {recognized_user}")
            message = f"{recognized_user} has already checked in today."
            raise HTTPException(status_code=400, detail=f"{recognized_user} has already checked in today")

        # Insert attendance record
        db_connection['attendance_collection'].insert_one({
            "user_id": user_details["user_id"],
            "date": today,
            "check_in": now_time,
            "check_out": None,
            "check_in_timestamp": now_full,
            "confidence": 1-min_distance
        })

        success_message = f"Hello, {recognized_user}. You have been checked in."
        logger.info(f"[{request_id}] {recognized_user} checked in at {now_time}")
        return {
            "message": f"{recognized_user} checked in at {now_time}",
            "name": recognized_user,
            "time": now_time,
            "confidence": 1-min_distance
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in check_in: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during check-in: {str(e)}")

@app.post("/check-out")
async def check_out(data: ImageData, request: Request):
    """Register user check-out attendance"""
    request_id = request.state.request_id
    try:
        if db_connection is None:
            raise HTTPException(status_code=503, detail="Database connection failed")

        embedding = extract_embedding(data.image, request_id)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected")

        users = db_connection['users_collection'].find({}, {"user_id":1, "name": 1, "embeddings": 1})
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
            logger.warning(f"[{request_id}] Unrecognized user attempted check-out")
            raise HTTPException(status_code=400, detail="User not recognized")

        today = datetime.date.today().strftime("%Y-%m-%d")
        now_time = datetime.datetime.now().strftime("%H:%M:%S")
        now_full = datetime.datetime.now()

        existing_record = db_connection['attendance_collection'].find_one({
            "user_id": user_details["user_id"],
            "date": today,
            "check_in": {"$exists": True} 
        })

        if not existing_record:
            message = f"{recognized_user} has not checked in today."
            logger.warning(f"[{request_id}] {message}")
            raise HTTPException(status_code=400, detail=f"{recognized_user} has not checked in today")

        if existing_record.get("check_out"):
            message = f"{recognized_user} has already checked out today."
            logger.warning(f"[{request_id}] {message}")
            raise HTTPException(status_code=400, detail=message)

        # Update attendance record with check-out time
        db_connection['attendance_collection'].update_one(
            {"_id": existing_record["_id"]},
            {"$set": {
                "check_out": now_time,
                "check_out_timestamp": now_full,
                "checkout_confidence": 1-min_distance
            }}
        )

        success_message = f"Thank you, {recognized_user}. You have been checked out."
        logger.info(f"[{request_id}] {recognized_user} checked out at {now_time}")
        return {
            "message": f"{recognized_user} checked out at {now_time}",
            "name": recognized_user,
            "time": now_time,
            "confidence": 1-min_distance
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in check_out: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during check-out: {str(e)}")

@app.get("/active_users")
async def get_active_users(
    department: Optional[str] = Query(None), 
    designation: Optional[str] = Query(None),
    request: Request = None
):
    """Get list of active users with optional filtering"""
    request_id = request.state.request_id if request else "system"
    try:
        # Ensure users_collection exists
        if db_connection is None:
            raise HTTPException(status_code=503, detail="Database connection error")

        query = {}
        if department:
            query["department"] = department
        if designation:
            query["designation"] = designation

        users_cursor = db_connection["users_collection"].find(
            query, 
            {"_id": 0, "user_id": 1, "name": 1, "email": 1, "department": 1, "designation": 1}
        )
        users = list(users_cursor)

        if not users:
            logger.info(f"[{request_id}] No active users found with filters: {query}")
            return {"message": "No active users", "users": []}

        logger.info(f"[{request_id}] Found {len(users)} active users")
        return {"active_users": users, "count": len(users)}

    except PyMongoError as e:
        logger.error(f"[{request_id}] Database error in get_active_users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in get_active_users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/get_users_check")
def get_users(request: Request):
    """Get all users with basic information"""
    request_id = request.state.request_id
    if db_connection is None:
        raise HTTPException(status_code=503, detail="Database connection failed")
        
    users = list(db_connection["users_collection"].find(
        {}, 
        {"_id": 0, "user_id": 1, "name": 1, "email": 1}
    ))
    logger.info(f"[{request_id}] Retrieved {len(users)} users")
    return {"users": json.loads(dumps(users)), "count": len(users)}

@app.get("/get_users")
def get_users_id(request: Request):
    """Get all user IDs and names"""
    request_id = request.state.request_id
    if db_connection is None:
        raise HTTPException(status_code=503, detail="Database connection failed")
        
    users = list(db_connection["users_collection"].find({}, {"_id": 0, "user_id": 1, "name": 1}))
    logger.info(f"[{request_id}] Retrieved {len(users)} user IDs")
    return json.loads(dumps(users))

@app.get("/get-attendance/{user_id}")
def get_attendance(user_id: int, request: Request):
    """Get attendance records for a specific user"""
    request_id = request.state.request_id
    if db_connection is None:
        raise HTTPException(status_code=503, detail="Database connection failed")
        
    try:
        # Validate user exists
        user = db_connection["users_collection"].find_one({"user_id": user_id})
        if not user:
            logger.warning(f"[{request_id}] Attendance request for non-existent user ID: {user_id}")
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
            
        attendance_records = list(db_connection["attendance_collection"].find(
            {"user_id": user_id}, 
            {"_id": 0, "date": 1, "check_in": 1, "check_out": 1}
        ))
        
        logger.info(f"[{request_id}] Retrieved {len(attendance_records)} attendance records for user {user_id}")
        return json.loads(dumps(attendance_records))
    except Exception as e:
        logger.error(f"[{request_id}] Error retrieving attendance for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving attendance data: {str(e)}")

# Request statistics (optional)
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "start_time": datetime.datetime.now()
}

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware to track request statistics"""
    request_stats["total_requests"] += 1
    
    response = await call_next(request)
    
    if response.status_code < 400:
        request_stats["successful_requests"] += 1
    else:
        request_stats["failed_requests"] += 1
        
    return response

@app.get("/metrics")
async def get_metrics():
    """Get API usage metrics (admin only)"""
    # In production, this should be protected
    uptime = (datetime.datetime.now() - request_stats["start_time"]).total_seconds()
    
    return {
        "total_requests": request_stats["total_requests"],
        "successful_requests": request_stats["successful_requests"],
        "failed_requests": request_stats["failed_requests"],
        "uptime_seconds": uptime,
        "request_success_rate": (
            request_stats["successful_requests"] / request_stats["total_requests"] * 100 
            if request_stats["total_requests"] > 0 else 0
        )
    }

# Graceful shutdown handler
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup resources on application shutdown"""
    logger.info("Application shutting down")
    if db_connection and "client" in db_connection:
        try:
            db_connection["client"].close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

if __name__ == "__main__":
    print("\n\n!!!!!!!!!! PYTHON SERVER IS UP !!!!!!!!!!!!\n\n")
    
    # Fixed: Use the app object directly for consistency
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,  # Set to False in production
        workers=int(os.getenv("WORKERS", 1))
    )