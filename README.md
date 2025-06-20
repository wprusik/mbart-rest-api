# BART Translator REST API

The program launches the REST API to the mBART interpreter (uses the mbart-large-50 model), containing the following endpoints:

```
GET    /              healthcheck
GET    /languages     list of languages
POST   /translate     text translation
```

# Running on docker
### 1. Go to one of the directories: gpu or cpu
Go to directory **src/gpu** or **src/cpu**, depending on what you want to run the model on.

### 2. Build image
```
docker build -t mbart-restapi:gpu .
```

### 3. Run container
```
docker run -d --name mBART_gpu -p 8000:80 mbart-restapi:gpu
```

Then go to Swagger at [http://localhost:8000/docs](http://localhost:8000/docs).
