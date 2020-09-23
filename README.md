## This application uses [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) to serve an image classifier model 

-----------
To run this application, clone the repo:

```bash
 git clone https://github.com/risenW/tensorflow_serving_app.git
 cd tensorflow_serving_app
 pip install -r requirements.txt
```

Install Tensorflow Serving
- [Link](https://www.tensorflow.org/tfx/serving/setup)


Start TF serving server
```
 docker run -p 8501:8501 --name tfserving_classifier \
 --mount type=bind,source=tf-server/img_classifier/,target=/models/img_classifier \
 -e MODEL_NAME=img_classifier -t tensorflow/serving
```

To make predictions, send HTTP request to server from another terminal.

```
  python predict.py
```

**Extra:**

To retrain the model, run the `python model.py`
