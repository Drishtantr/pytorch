import jwt
import datetime

SECRET_KEY = "apple"


def get_token():
    payload = {
        "user": "alice",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return {"token": token}


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        print("Valid token, payload:", payload)
        return True
    except jwt.ExpiredSignatureError:
        print("Token expired")
    except jwt.InvalidTokenError:
        print("Invalid token")
    return False


# Example usage
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYWxpY2UiLCJleHAiOjE3NTg3NTg1NDB9.h68jIp36dDter6pj6PIuDQpt-V7T5whdoONozg91F5M"
verify_token(token)  # Should return True
