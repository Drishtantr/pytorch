from fastapi import FastAPI, Request, HTTPException, Depends
import jwt
import datetime

app = FastAPI()
SECRET_KEY = "apple"


@app.get("/get-token")
def get_token():
    payload = {
        "user": "alice",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5)  # expires in 5 min
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return {"token": token}


def verify_jwt(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    print(auth_header)

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload   # you can pass this into endpoints
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/secure")
def secure_route(payload=Depends(verify_jwt)):
    return {"message": "Access granted", "payload": payload}


@app.get("/")
def public_route():
    return {"message": "This is public"}
