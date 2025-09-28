import jwt
import datetime

SECRET_KEY = "apple"

payload = {
    "agent": "alice",
    "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5)  # expires in 5 min
}

token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
print(token)
