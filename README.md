# From Bytes to Bites: AI-Driven Calorie Detection in the Palm of Your Hand

This repository hosts a Flask API for processing images using a YOLO model to detect calories. This API is used in conjunction with the Android application in the [From-Bytes-to-Bites-AI-Driven-Calorie-Detection-in-the-Palm-of-Your-Hand](https://github.com/Firdausfatihul/From-Bytes-to-Bites-AI-Driven-Calorie-Detection-in-the-Palm-of-Your-Hand) repository, specifically in the `com.example.calorificomputervision.ui.pages` package.

## Features

- Upload images via HTTP POST request.
- Process images using a pre-trained YOLO model.
- Return the processed images with detected objects.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/Firdausfatihul/bytestobyte_api.git
   cd bytestobyte_api
   ```

2. **Create a virtual environment:**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**

   ```sh
   python app.py
   ```

### Running the API

Ensure the Flask application is running on your local machine. By default, it will be accessible at `http://127.0.0.1:5000`.

### Using Ngrok for Public Access

To make your Flask API publicly accessible, you can use Ngrok:

1. **Download and install Ngrok:** Follow the instructions at [ngrok.com](https://ngrok.com/download).

2. **Expose your local server:** Run the following command to start Ngrok:

   ```sh
   ngrok http 5000
   ```

3. **Get the public URL:** Ngrok will provide a forwarding URL (e.g., `http://1234abcd.ngrok.io`). Use this URL to access your API from the Android application or any other client.

## Usage

### Endpoint: `/process_image`

- **Method:** `POST`
- **Description:** Upload an image to be processed by the YOLO model.
- **Request Parameters:**
  - `file`: The image file to be uploaded (must be in `png`, `jpg`, or `jpeg` format).

- **Response:**
  - `200 OK`: JSON response containing the URL of the processed image.
  - `400 Bad Request`: JSON response containing an error message.

#### Example Usage with `curl`

```sh
curl -X POST -F "file=@path/to/your/image.jpg" http://127.0.0.1:5000/process_image
```

## Integration with Android

This API is integrated into the Android application found in the [From-Bytes-to-Bites-AI-Driven-Calorie-Detection-in-the-Palm-of-Your-Hand](https://github.com/Firdausfatihul/From-Bytes-to-Bites-AI-Driven-Calorie-Detection-in-the-Palm-of-Your-Hand) repository, particularly in the `com.example.calorificomputervision.ui.pages` package.

### Android Integration Example

Here is a brief outline of how to integrate the API with your Android application:

1. **Capture Image:** Use `CameraX` to capture an image and save it to the device.

2. **Upload Image:** Create an HTTP POST request to upload the image to the Flask API.

3. **Process Response:** Receive and display the processed image in your application.

#### Ngrok Integration in Android

When using Ngrok, update your API base URL in the Android application to the Ngrok forwarding URL:

```kotlin
val apiUrl = "http://1234abcd.ngrok.io/process_image"
```

This allows the Android application to communicate with your local Flask server over the internet.
