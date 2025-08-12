import requests

# 單張預測
with open('captcha_00003.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())

with open('captcha_00005.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())

with open('captcha_00007.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())