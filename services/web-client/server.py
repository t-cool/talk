import os
import uuid
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from livekit import api

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-client-server")

app = FastAPI(title="Voice Assistant Web Client Server")

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secretkeyforlocalvoiceai2026")
LIVEKIT_PUBLIC_URL = os.getenv("LIVEKIT_PUBLIC_URL", "ws://localhost:7880")

@app.get("/api/token")
async def get_token(
    room: str = Query("voice-room-1"),
    participant: str = Query(None),
):
    identity = participant or f"user-{uuid.uuid4().hex[:6]}"
    name = f"Participant-{identity[-4:]}"

    try:
        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )

        return JSONResponse(
            content={
                "token": token,
                "url": LIVEKIT_PUBLIC_URL,
                "room": room,
                "identity": identity,
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    return JSONResponse(content={"livekitUrl": LIVEKIT_PUBLIC_URL})

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
