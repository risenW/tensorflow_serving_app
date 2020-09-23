import matplotlib.pyplot as plt
import requests
import base64
import json
import numpy as np
from tensorflow.keras.datasets.mnist import load_data


#load MNIST dataset
(_, _), (x_test, y_test) = load_data()

# reshape data to have a single channel
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
# normalize pixel values
x_test = x_test.astype('float32') / 255.0

#server URL
url = 'http://localhost:8501/v1/models/img_classifier:predict'


def show(idx, title):
  plt.figure()
  plt.imshow(x_test[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})
  plt.show()

def make_prediction(instances):
    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions
    


predictions = make_prediction(x_test[0:4])

for i, pred in enumerate(predictions):
  print(f"True Value: {y_test[i]}, Predicted Value: {np.argmax(pred)}")

show(0, 'The model predicted {}, and it was actually {}'.format(np.argmax(predictions[0]), y_test[0]))