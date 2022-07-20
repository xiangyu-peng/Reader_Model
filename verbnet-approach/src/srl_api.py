import requests
import json
import random
import time

def call_verbatlas(sentence):
    url = "http://localhost:5000/"
    # print(' sentence.encode',  sentence.encode('utf-8'))
    response = requests.request("POST", url, data=sentence.encode('utf-8'))
    # print('response,', response)
    response = json.JSONDecoder().decode(response.text)

    return response

if  __name__ == '__main__':
    print(call_verbatlas('David drove to work'))
