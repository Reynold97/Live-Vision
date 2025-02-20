Explore vision capabilities with the Gemini API

Python Node.js Go REST

Try a Colab notebook
View notebook on GitHub
Gemini models are able to process images and videos, enabling many frontier developer use cases that would have historically required domain specific models. Some of Gemini's vision capabilities include the ability to:

Caption and answer questions about images
Transcribe and reason over PDFs, including up to 2 million tokens
Describe, segment, and extract information from videos up to 90 minutes long
Detect objects in an image and return bounding box coordinates for them
Gemini was built to be multimodal from the ground up and we continue to push the frontier of what is possible.

Image input
For total image payload size less than 20MB, we recommend either uploading base64 encoded images or directly uploading locally stored image files.

Working with local images
If you are using the Python imaging library (Pillow ), you can use PIL image objects too.


from google import genai
from google.genai import types

import PIL.Image

image = PIL.Image.open('/path/to/image.png')

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["What is this image?", image])

print(response.text)
Base64 encoded images
You can upload public image URLs by encoding them as Base64 payloads. The following code example shows how to do this using only standard library tools:


from google import genai
from google.genai import types

import requests

image_path = "https://goo.gle/instrument-img"
image = requests.get(image_path)

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What is this image?",
              types.Part.from_bytes(data=image.content, mime_type="image/jpeg")])

print(response.text)
Multiple images
To prompt with multiple images, you can provide multiple images in the call to generate_content. These can be in any supported format, including base64 or PIL.


from google import genai
from google.genai import types

import pathlib
import PIL.Image

image_path_1 = "path/to/your/image1.jpeg"  # Replace with the actual path to your first image
image_path_2 = "path/to/your/image2.jpeg" # Replace with the actual path to your second image

image_url_1 = "https://goo.gle/instrument-img" # Replace with the actual URL to your third image

pil_image = PIL.Image.open(image_path_1)

b64_image = types.Part.from_bytes(
    data=pathlib.Path(image_path_2).read_bytes(),
    mime_type="image/jpeg"
)

downloaded_image = requests.get(image_url_1)

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What do these images have in common?",
              pil_image, b64_image, downloaded_image])

print(response.text)
Note that these inline data calls don't include many of the features available through the File API, such as getting file metadata, listing, or deleting files.

Large image payloads
When the combination of files and system instructions that you intend to send is larger than 20 MB in size, use the File API to upload those files.

Use the media.upload method of the File API to upload an image of any size.

Note: The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but cannot be downloaded from the API. It is available at no cost in all regions where the Gemini API is available.
After uploading the file, you can make GenerateContent requests that reference the File API URI. Select the generative model and provide it with a text prompt and the uploaded image.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

img_path = "/path/to/Cajun_instruments.jpg"
file_ref = client.files.upload(file=img_path)
print(f'{file_ref=}')

client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["What can you tell me about these instruments?",
              file_ref])

print(response.text)
OpenAI Compatibility
You can access Gemini's image understanding capabilities using the OpenAI libraries. This lets you integrate Gemini into existing OpenAI workflows by updating three lines of code and using your Gemini API key. See the Image understanding example for code demonstrating how to send images encoded as Base64 payloads.

Prompting with images
In this tutorial, you will upload images using the File API or as inline data and generate content based on those images.

Technical details (images)
Gemini 1.5 Pro and 1.5 Flash support a maximum of 3,600 image files.

Images must be in one of the following image data MIME types:

PNG - image/png
JPEG - image/jpeg
WEBP - image/webp
HEIC - image/heic
HEIF - image/heif
Tokens
Here's how tokens are calculated for images:

Gemini 1.0 Pro Vision: Each image accounts for 258 tokens.
Gemini 1.5 Flash and Gemini 1.5 Pro: If both dimensions of an image are less than or equal to 384 pixels, then 258 tokens are used. If one dimension of an image is greater than 384 pixels, then the image is cropped into tiles. Each tile size defaults to the smallest dimension (width or height) divided by 1.5. If necessary, each tile is adjusted so that it's not smaller than 256 pixels and not greater than 768 pixels. Each tile is then resized to 768x768 and uses 258 tokens.
Gemini 2.0 Flash: Image inputs with both dimensions <=384 pixels are counted as 258 tokens. Images larger in one or both dimensions are cropped and scaled as needed into tiles of 768x768 pixels, each counted as 258 tokens.
For best results
Rotate images to the correct orientation before uploading.
Avoid blurry images.
If using a single image, place the text prompt after the image.
Capabilities
This section outlines specific vision capabilities of the Gemini model, including object detection and bounding box coordinates.

Get a bounding box for an object
Gemini models are trained to return bounding box coordinates as relative widths or heights in the range of [0, 1]. These values are then scaled by 1000 and converted to integers. Effectively, the coordinates represent the bounding box on a 1000x1000 pixel version of the image. Therefore, you'll need to convert these coordinates back to the dimensions of your original image to accurately map the bounding boxes.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

prompt = (
  "Return a bounding box for each of the objects in this image "
  "in [ymin, xmin, ymax, xmax] format.")

response = client.models.generate_content(
  model="gemini-1.5-pro",
  contents=[sample_file_1, prompt])

print(response.text)
You can use bounding boxes for object detection and localization within images and video. By accurately identifying and delineating objects with bounding boxes, you can unlock a wide range of applications and enhance the intelligence of your projects.

Key Benefits
Simple: Integrate object detection capabilities into your applications with ease, regardless of your computer vision expertise.
Customizable: Produce bounding boxes based on custom instructions (e.g. "I want to see bounding boxes of all the green objects in this image"), without having to train a custom model.
Technical Details
Input: Your prompt and associated images or video frames.
Output: Bounding boxes in the [y_min, x_min, y_max, x_max] format. The top left corner is the origin. The x and y axis go horizontally and vertically, respectively. Coordinate values are normalized to 0-1000 for every image.
Visualization: AI Studio users will see bounding boxes plotted within the UI.
For Python developers, try the 2D spatial understanding notebook or the experimental 3D pointing notebook.

Normalize coordinates
The model returns bounding box coordinates in the format [y_min, x_min, y_max, x_max]. To convert these normalized coordinates to the pixel coordinates of your original image, follow these steps:

Divide each output coordinate by 1000.
Multiply the x-coordinates by the original image width.
Multiply the y-coordinates by the original image height.
To explore more detailed examples of generating bounding box coordinates and visualizing them on images, we encourage you to review our Object Detection cookbook example.

Prompting with video
In this tutorial, you will upload a video using the File API and generate content based on those images.

Note: The File API is required to upload video files, due to their size. However, the File API is only available for Python, Node.js, Go, and REST.
Technical details (video)
Gemini 1.5 Pro and Flash support up to approximately an hour of video data.

Video must be in one of the following video format MIME types:

video/mp4
video/mpeg
video/mov
video/avi
video/x-flv
video/mpg
video/webm
video/wmv
video/3gpp
The File API service extracts image frames from videos at 1 frame per second (FPS) and audio at 1Kbps, single channel, adding timestamps every second. These rates are subject to change in the future for improvements in inference.

Note: The details of fast action sequences may be lost at the 1 FPS frame sampling rate. Consider slowing down high-speed clips for improved inference quality.
Individual frames are 258 tokens, and audio is 32 tokens per second. With metadata, each second of video becomes ~300 tokens, which means a 1M context window can fit slightly less than an hour of video.

To ask questions about time-stamped locations, use the format MM:SS, where the first two digits represent minutes and the last two digits represent seconds.

For best results:

Use one video per prompt.
If using a single video, place the text prompt after the video.
Upload a video file using the File API
Note: The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but they cannot be downloaded using any API. It is available at no cost in all regions where the Gemini API is available.
The File API accepts video file formats directly. This example uses the short NASA film "Jupiter's Great Red Spot Shrinks and Grows". Credit: Goddard Space Flight Center (GSFC)/David Ladd (2018).

"Jupiter's Great Red Spot Shrinks and Grows" is in the public domain and does not show identifiable people. (NASA image and media usage guidelines.)

Start by retrieving the short video:


wget https://storage.googleapis.com/generativeai-downloads/images/GreatRedSpot.mp4
Upload the video using the File API and print the URI.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

print("Uploading file...")
video_file = client.files.upload(file="GreatRedSpot.mp4")
print(f"Completed upload: {video_file.uri}")
Verify file upload and check state
Verify the API has successfully received the files by calling the files.get method.

Note: Video files have a State field in the File API. When a video is uploaded, it will be in the PROCESSING state until it is ready for inference. Only ACTIVE files can be used for model inference.

import time

# Check whether the file is ready to be used.
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(1)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)

print('Done')
Prompt with a video and text
Once the uploaded video is in the ACTIVE state, you can make GenerateContent requests that specify the File API URI for that video. Select the generative model and provide it with the uploaded video and a text prompt.


from IPython.display import Markdown

# Pass the video file reference like any other media part.
response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[
        video_file,
        "Summarize this video. Then create a quiz with answer key "
        "based on the information in the video."])

# Print the response, rendering any Markdown
Markdown(response.text)
Refer to timestamps in the content
You can use timestamps of the form HH:MM:SS to refer to specific moments in the video.


prompt = "What are the examples given at 01:05 and 01:19 supposed to show us?"

response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[video_file, prompt])

print(response.text)
Transcribe video and provide visual descriptions
The Gemini models can transcribe and provide visual descriptions of video content by processing both the audio track and visual frames. For visual descriptions, the model samples the video at a rate of 1 frame per second. This sampling rate may affect the level of detail in the descriptions, particularly for videos with rapidly changing visuals.


prompt = (
    "Transcribe the audio from this video, giving timestamps for "
    "salient events in the video. Also provide visual descriptions.")

response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[video_file, prompt])

print(response.text)
List files
You can list all files uploaded using the File API and their URIs using files.list.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

print('My files:')
for f in client.files.list():
  print(" ", f'{f.name}: {f.uri}')
Delete files
Files uploaded using the File API are automatically deleted after 2 days. You can also manually delete them using files.delete.


from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

# Upload a file
poem_file = client.files.upload(file="poem.txt")

# Files will auto-delete after a period.
print(poem_file.expiration_time)

# Or they can be deleted explicitly.
dr = client.files.delete(name=poem_file.name)

try:
  client.models.generate_content(
      model="gemini-2.0-flash-exp",
      contents=['Finish this poem:', poem_file])
except genai.errors.ClientError as e:
  print(e.code)  # 403
  print(e.status)  # PERMISSION_DENIED
  print(e.message)  # You do not have permission to access the File .. or it may not exist.



Gemini Vision Notebook:


{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Tce3stUlHN0L"
      },
      "source": [
        "##### Copyright 2024 Google LLC."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "cellView": "form",
        "id": "tuOe1ymfHZPu"
      },
      "outputs": [],
      "source": [
        "#@title Licensed under the Apache License, Version 2.0 (the \"License\");\n",
        "# you may not use this file except in compliance with the License.\n",
        "# You may obtain a copy of the License at\n",
        "#\n",
        "# https://www.apache.org/licenses/LICENSE-2.0\n",
        "#\n",
        "# Unless required by applicable law or agreed to in writing, software\n",
        "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
        "# See the License for the specific language governing permissions and\n",
        "# limitations under the License."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "084u8u0DpBlo"
      },
      "source": [
        "# Explore vision capabilities with the Gemini API"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZFWzQEqNosrS"
      },
      "source": [
        "<table class=\"tfo-notebook-buttons\" align=\"left\">\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://ai.google.dev/gemini-api/docs/vision\"><img src=\"https://ai.google.dev/static/site-assets/images/docs/notebook-site-button.png\" height=\"32\" width=\"32\" />View on ai.google.dev</a>\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemini-api/docs/vision.ipynb\"><img src=\"https://www.tensorflow.org/images/colab_logo_32px.png\" />Run in Google Colab</a>\n",
        "  </td>\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/docs/vision.ipynb\"><img src=\"https://www.tensorflow.org/images/GitHub-Mark-32px.png\" />View source on GitHub</a>\n",
        "  </td>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3c5e92a74e64"
      },
      "source": [
        "The Gemini API is able to process images and videos, enabling a multitude of\n",
        " exciting developer use cases. Some of Gemini's vision capabilities include\n",
        " the ability to:\n",
        "\n",
        "*   Caption and answer questions about images\n",
        "*   Transcribe and reason over PDFs, including long documents up to 2 million token context window\n",
        "*   Describe, segment, and extract information from videos,\n",
        "including both visual frames and audio, up to 90 minutes long\n",
        "*   Detect objects in an image and return bounding box coordinates for them\n",
        "\n",
        "This tutorial demonstrates some possible ways to prompt the Gemini API with\n",
        "images and video input, provides code examples,\n",
        "and outlines prompting best practices with multimodal vision capabilities.\n",
        "All output is text-only."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VxCstRHvpX0r"
      },
      "source": [
        "## Setup\n",
        "\n",
        "Before you use the File API, you need to install the Gemini API SDK package and configure an API key. This section describes how to complete these setup steps."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "G6J_rV2ipmj_"
      },
      "source": [
        "### Install the Python SDK and import packages\n",
        "\n",
        "The Python SDK for the Gemini API is contained in the [google-generativeai](https://pypi.org/project/google-generativeai/) package. Install the dependency using pip."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mN8x8DPgu9Kv"
      },
      "outputs": [],
      "source": [
        "!pip install -q -U google-generativeai"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NInUZ4hwDq6d"
      },
      "source": [
        "Import the necessary packages."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0x3xmmWrDtEH"
      },
      "outputs": [],
      "source": [
        "import google.generativeai as genai\n",
        "from IPython.display import Markdown"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "l8g4hTRotheH"
      },
      "source": [
        "### Set up your API key\n",
        "\n",
        "The File API uses API keys for authentication and access. Uploaded files are associated with the project linked to the API key. Unlike other Gemini APIs that use API keys, your API key also grants access to data you've uploaded to the File API, so take extra care in keeping your API key secure. For more on keeping your keys\n",
        "secure, see [Best practices for using API\n",
        "keys](https://support.google.com/googleapi/answer/6310037).\n",
        "\n",
        "Store your API key in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or are unfamiliar with Colab Secrets, refer to the [Authentication quickstart](https://github.com/google-gemini/gemini-api-cookbook/blob/main/quickstarts/Authentication.ipynb)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d6lYXRcjthKV"
      },
      "outputs": [],
      "source": [
        "from google.colab import userdata\n",
        "GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')\n",
        "\n",
        "genai.configure(api_key=GOOGLE_API_KEY)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "c-z4zsCUlaru"
      },
      "source": [
        "## Prompting with images\n",
        "\n",
        "In this tutorial, you will upload images using the File API or as inline data and generate content based on those images."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AKNehP2tr3Cr"
      },
      "source": [
        "### Technical details (images)\n",
        "Gemini 1.5 Pro and Flash support a maximum of 3,600 image files.\n",
        "\n",
        "Images must be in one of the following image data [MIME types](https://developers.google.com/drive/api/guides/ref-export-formats):\n",
        "\n",
        "-   PNG - `image/png`\n",
        "-   JPEG - `image/jpeg`\n",
        "-   WEBP - `image/webp`\n",
        "-   HEIC - `image/heic`\n",
        "-   HEIF - `image/heif`\n",
        "\n",
        "Each image is equivalent to 258 tokens.\n",
        "\n",
        "While there are no specific limits to the number of pixels in an image besides the model’s context window, larger images are scaled down to a maximum resolution of 3072x3072 while preserving their original aspect ratio, while smaller images are scaled up to 768x768 pixels. There is no cost reduction for images at lower sizes, other than bandwidth, or performance improvement for images at higher resolution.\n",
        "\n",
        "For best results:\n",
        "\n",
        "*   Rotate images to the correct orientation before uploading.\n",
        "*   Avoid blurry images.\n",
        "*   If using a single image, place the text prompt after the image."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2fa34d5c0db8"
      },
      "source": [
        "## Image input\n",
        "\n",
        "For total image payload size less than 20MB, it's recommended to either upload\n",
        "base64 encoded images or directly upload locally stored image files."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8336e412da3e"
      },
      "source": [
        "### Base64 encoded images\n",
        "\n",
        "You can upload public image URLs by encoding them as Base64 payloads.\n",
        "You can use the httpx library to fetch the image URLs.\n",
        "The following code example shows how to do this:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aa9a0e452544"
      },
      "outputs": [],
      "source": [
        "import httpx\n",
        "import base64\n",
        "\n",
        "# Retrieve an image\n",
        "image_path = \"https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg/2560px-Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg\"\n",
        "image = httpx.get(image_path)\n",
        "\n",
        "# Choose a Gemini model\n",
        "model = genai.GenerativeModel(model_name=\"gemini-1.5-pro\")\n",
        "\n",
        "# Create a prompt\n",
        "prompt = \"Caption this image.\"\n",
        "response = model.generate_content(\n",
        "    [\n",
        "        {\n",
        "            \"mime_type\": \"image/jpeg\",\n",
        "            \"data\": base64.b64encode(image.content).decode(\"utf-8\"),\n",
        "        },\n",
        "        prompt,\n",
        "    ]\n",
        ")\n",
        "\n",
        "Markdown(\">\" + response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "f47333dabe62"
      },
      "source": [
        "### Multiple images\n",
        "\n",
        "To prompt with multiple images in Base64 encoded format, you can do the\n",
        "following:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2816ea3d2d91"
      },
      "outputs": [],
      "source": [
        "import httpx\n",
        "import base64\n",
        "\n",
        "# Retrieve two images\n",
        "image_path_1 = \"https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg/2560px-Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg\"\n",
        "image_path_2 = \"https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg\"\n",
        "\n",
        "image_1 = httpx.get(image_path_1)\n",
        "image_2 = httpx.get(image_path_2)\n",
        "\n",
        "# Create a prompt\n",
        "prompt = \"Generate a list of all the objects contained in both images.\"\n",
        "\n",
        "response = model.generate_content([\n",
        "{'mime_type':'image/jpeg', 'data': base64.b64encode(image_1.content).decode('utf-8')},\n",
        "{'mime_type':'image/jpeg', 'data': base64.b64encode(image_2.content).decode('utf-8')}, prompt])\n",
        "\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Lm862F3zthiI"
      },
      "source": [
        "### Upload one or more locally stored image files\n",
        "\n",
        "Alternatively, you can upload one or more locally stored image files.. \n",
        "\n",
        "You can download and use our drawings of [piranha-infested waters](https://storage.googleapis.com/generativeai-downloads/images/piranha.jpg) and a [firefighter with a cat](https://storage.googleapis.com/generativeai-downloads/images/firefighter.jpg). First, save these files to your local directory.\n",
        "\n",
        "Then click **Files** on the left sidebar. For each file, click the **Upload** button, then navigate to that file's location and upload it:\n",
        "\n",
        "<img width=400 src=\"https://ai.google.dev/tutorials/images/colab_upload.png\">\n",
        "\n",
        "When the combination of files and system instructions that you intend to send is larger than 20 MB in size, use the File API to upload those files. Smaller files can instead be called locally from the Gemini API:\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "XzMhQ8MWub5_"
      },
      "outputs": [],
      "source": [
        "import PIL.Image\n",
        "\n",
        "sample_file_2 = PIL.Image.open('piranha.jpg')\n",
        "sample_file_3 = PIL.Image.open('firefighter.jpg')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "da11223550a9"
      },
      "outputs": [],
      "source": [
        "import google.generativeai as genai\n",
        "\n",
        "# Choose a Gemini model.\n",
        "model = genai.GenerativeModel(model_name=\"gemini-1.5-pro-latest\")\n",
        "\n",
        "# Create a prompt.\n",
        "prompt = \"Write an advertising jingle based on the items in both images.\"\n",
        "\n",
        "response = model.generate_content([sample_file_2, sample_file_3, prompt])\n",
        "\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "736c83de95a1"
      },
      "source": [
        "Note that these inline data calls don't include many of the features available\n",
        "through the File API, such as getting file metadata,\n",
        "[listing](https://ai.google.dev/gemini-api/docs/vision?lang=python#list-files),\n",
        "or [deleting files](https://ai.google.dev/gemini-api/docs/vision?lang=python#delete-files)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0d6f7af7c2ff"
      },
      "source": [
        "### Large image payloads"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rsdNkDszLBmQ"
      },
      "source": [
        "#### Upload an image file using the File API\n",
        "\n",
        "When the combination of files and system instructions that you intend to send is larger than 20 MB in size, use the File API to upload those files.\n",
        "\n",
        "**NOTE**: The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but cannot be downloaded from the API. It is available at no cost in all regions where the Gemini API is available."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qfa2VSqEsulq"
      },
      "source": [
        "Upload the image using [`media.upload`](https://ai.google.dev/api/rest/v1beta/media/upload) and print the URI, which is used as a reference in Gemini API calls."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2e9f9469b337"
      },
      "outputs": [],
      "source": [
        "!curl -o jetpack.jpg https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "N9NxXGZKKusG"
      },
      "outputs": [],
      "source": [
        "# Upload the file and print a confirmation.\n",
        "sample_file = genai.upload_file(path=\"jetpack.jpg\",\n",
        "                            display_name=\"Jetpack drawing\")\n",
        "\n",
        "print(f\"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cto22vhKOvGQ"
      },
      "source": [
        "The `response` shows that the File API stored the specified `display_name` for the uploaded file and a `uri` to reference the file in Gemini API calls. Use `response` to track how uploaded files are mapped to URIs.\n",
        "\n",
        "Depending on your use case, you can also store the URIs in structures such as a `dict` or a database."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ds5iJlaembWe"
      },
      "source": [
        "#### Verify image file upload and get metadata\n",
        "\n",
        "You can verify the API successfully stored the uploaded file and get its metadata by calling [`files.get`](https://ai.google.dev/api/rest/v1beta/files/get) through the SDK. Only the `name` (and by extension, the `uri`) are unique. Use `display_name` to identify files only if you manage uniqueness yourself."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kLFsVLFHOWSV"
      },
      "outputs": [],
      "source": [
        "file = genai.get_file(name=sample_file.name)\n",
        "print(f\"Retrieved file '{file.display_name}' as: {sample_file.uri}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BqzIGKBmnFoJ"
      },
      "source": [
        "Depending on your use case, you can store the URIs in structures, such as a `dict` or a database."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EPPOECHzsIGJ"
      },
      "source": [
        "#### Prompt with the uploaded image and text\n",
        "\n",
        "After uploading the file, you can make GenerateContent requests that reference the File API URI. Select the generative model and provide it with a text prompt and the uploaded image."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZYVFqmLkl5nE"
      },
      "outputs": [],
      "source": [
        "# Choose a Gemini model.\n",
        "model = genai.GenerativeModel(model_name=\"gemini-1.5-pro-latest\")\n",
        "\n",
        "# Prompt the model with text and the previously uploaded image.\n",
        "response = model.generate_content([sample_file, \"Describe how this product might be manufactured.\"])\n",
        "\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "c63c1a7a8e32"
      },
      "source": [
        "## Capabilties\n",
        "\n",
        "This section outlines specific vision capabilities of the Gemini model, including object detection and bounding box coordinates."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7e16d742407a"
      },
      "source": [
        "### Get bounding boxes\n",
        "\n",
        "Gemini models are trained to return bounding box coordinates as relative widths or heights in the range of [0, 1]. These values are then scaled by 1000 and converted to integers. Effectively, the coordinates represent the bounding box on a 1000x1000 pixel version of the image. Therefore, you'll need to convert these coordinates back to the dimensions of your original image to accurately map the bounding boxes."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "778dd36334f4"
      },
      "outputs": [],
      "source": [
        "# Choose a Gemini model.\n",
        "model = genai.GenerativeModel(model_name=\"gemini-1.5-pro-latest\")\n",
        "\n",
        "# Create a prompt to detect bounding boxes.\n",
        "prompt = \"Return a bounding box for each of the objects in this image in [ymin, xmin, ymax, xmax] format.\"\n",
        "response = model.generate_content([sample_file_2, prompt])\n",
        "\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "b8e422c55df2"
      },
      "source": [
        "The model returns bounding box coordinates in the format\n",
        "`[ymin, xmin, ymax, xmax]`. To convert these normalized coordinates\n",
        "to the pixel coordinates of your original image, follow these steps:\n",
        "\n",
        "1.    Divide each output coordinate by 1000.\n",
        "1.    Multiply the x-coordinates by the original image width.\n",
        "1.    Multiply the y-coordinates by the original image height.\n",
        "\n",
        "To explore more detailed examples of generating bounding box coordinates and\n",
        "visualizing them on images, review our [Object Detection cookbook example](https://github.com/google-gemini/cookbook/blob/main/examples/Object_detection.ipynb)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TaUZc1mvLkHY"
      },
      "source": [
        "## Prompting with video\n",
        "\n",
        "In this tutorial, you will upload a video using the File API and generate content based on those images."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nDN32NDPxXGX"
      },
      "source": [
        "### Technical details (video)\n",
        "\n",
        "Gemini 1.5 Pro and Flash support up to approximately an hour of video data.\n",
        "\n",
        "Video must be in one of the following video format [MIME types](https://developers.google.com/drive/api/guides/ref-export-formats):\n",
        "  -   `video/mp4`\n",
        "  -   `video/mpeg`\n",
        "  -   `video/mov`\n",
        "  -   `video/avi`\n",
        "  -   `video/x-flv`\n",
        "  -   `video/mpg`\n",
        "  -   `video/webm`\n",
        "  -   `video/wmv`\n",
        "  -   `video/3gpp`\n",
        "\n",
        "The File API service currently extracts image frames from videos at 1 frame per second (FPS) and audio at 1Kbps, single channel, adding timestamps every second. These rates are subject to change in the future for improvements in inference.\n",
        "\n",
        "**NOTE:** The finer details of fast action sequences may be lost at the 1 FPS frame sampling rate. Consider slowing down high-speed clips for improved inference quality.\n",
        "\n",
        "Individual frames are 258 tokens, and audio is 32 tokens per second. With metadata, each second of video becomes ~300 tokens, which means a 1M context window can fit slightly less than an hour of video.\n",
        "\n",
        "To ask questions about time-stamped locations, use the format `MM:SS`, where the first two digits represent minutes and the last two digits represent seconds.\n",
        "\n",
        "For best results:\n",
        "\n",
        "*   Use one video per prompt.\n",
        "*   If using a single video, place the text prompt after the video."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MNvhBdoDFnTC"
      },
      "source": [
        "### Upload a video file to the File API\n",
        "\n",
        "**NOTE**: The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but they cannot be downloaded using any API. It is available at no cost in all regions where the Gemini API is available.\n",
        "\n",
        "The File API accepts video file formats directly. This example uses the short NASA film [\"Jupiter's Great Red Spot Shrinks and Grows\"](https://www.youtube.com/watch?v=JDi4IdtvDVE0). Credit: Goddard Space Flight Center (GSFC)/David Ladd (2018).\n",
        "\n",
        "> \"Jupiter's Great Red Spot Shrinks and Grows\" is in the public domain and does not show identifiable people. ([NASA image and media usage guidelines.](https://www.nasa.gov/nasa-brand-center/images-and-media/))\n",
        "\n",
        "Start by retrieving the short video:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "V4XeFdX1rxaE"
      },
      "outputs": [],
      "source": [
        "!wget https://storage.googleapis.com/generativeai-downloads/images/GreatRedSpot.mp4"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZusSiIg2T6ls"
      },
      "source": [
        "Upload the video to the File API and print the URI."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_HzrDdp2Q1Cu"
      },
      "outputs": [],
      "source": [
        "video_file_name = \"GreatRedSpot.mp4\"\n",
        "\n",
        "print(f\"Uploading file...\")\n",
        "video_file = genai.upload_file(path=video_file_name)\n",
        "print(f\"Completed upload: {video_file.uri}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oOZmTUb4FWOa"
      },
      "source": [
        "### Verify file upload and check state\n",
        "\n",
        "Verify the API has successfully received the files by calling the [`files.get`](https://ai.google.dev/api/rest/v1beta/files/get) method.\n",
        "\n",
        "**NOTE**: Video files have a `State` field in the File API. When a video is uploaded, it will be in the `PROCESSING` state until it is ready for inference. Only `ACTIVE` files can be used for model inference."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SHMVCWHkFhJW"
      },
      "outputs": [],
      "source": [
        "import time\n",
        "\n",
        "# Check whether the file is ready to be used.\n",
        "while video_file.state.name == \"PROCESSING\":\n",
        "    print('.', end='')\n",
        "    time.sleep(10)\n",
        "    video_file = genai.get_file(video_file.name)\n",
        "\n",
        "if video_file.state.name == \"FAILED\":\n",
        "  raise ValueError(video_file.state.name)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IYIIHsvQt0_W"
      },
      "source": [
        "### Prompt with a video and text\n",
        "\n",
        "Once the uploaded video is in the `ACTIVE` state, you can make `GenerateContent` requests that specify the File API URI for that video. Select the generative model and provide it with the uploaded video and a text prompt."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sHH0ZR6Yt42S"
      },
      "outputs": [],
      "source": [
        "# Create the prompt.\n",
        "prompt = \"Summarize this video. Then create a quiz with answer key based on the information in the video.\"\n",
        "\n",
        "# Choose a Gemini model.\n",
        "model = genai.GenerativeModel(model_name=\"gemini-1.5-pro-latest\")\n",
        "\n",
        "# Make the LLM request.\n",
        "print(\"Making LLM inference request...\")\n",
        "response = model.generate_content([video_file, prompt],\n",
        "                                  request_options={\"timeout\": 600})\n",
        "\n",
        "# Print the response, rendering any Markdown\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zS5NmQeXLqeS"
      },
      "source": [
        "### Refer to timestamps in the content\n",
        "\n",
        "You can use timestamps of the form `MM:SS` to refer to specific moments in the video."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ypZuGQ-2LqeS"
      },
      "outputs": [],
      "source": [
        "# Create the prompt.\n",
        "prompt = \"What are the examples given at 01:05 and 01:19 supposed to show us?\"\n",
        "\n",
        "# Choose a Gemini model.\n",
        "model = genai.GenerativeModel(model_name=\"gemini-1.5-pro-latest\")\n",
        "\n",
        "# Make the LLM request.\n",
        "print(\"Making LLM inference request...\")\n",
        "response = model.generate_content([prompt, video_file],\n",
        "                                  request_options={\"timeout\": 600})\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JQE0XjgMZSJo"
      },
      "source": [
        "### Transcribe video and provide visual descriptions\n",
        "\n",
        "The Gemini models can transcribe and provide visual descriptions of video content\n",
        "by processing both the audio track and visual frames.\n",
        "For visual descriptions, the model samples the video at a rate of **1 frame\n",
        "per second**. This sampling rate may affect the level of detail in the\n",
        "descriptions, particularly for videos with rapidly changing visuals."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_JrcMsYnYXpJ"
      },
      "outputs": [],
      "source": [
        "# Create the prompt.\n",
        "prompt = \"Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions.\"\n",
        "\n",
        "# Choose a Gemini model.\n",
        "model = genai.GenerativeModel(model_name=\"gemini-1.5-pro-latest\")\n",
        "\n",
        "# Make the LLM request.\n",
        "print(\"Making LLM inference request...\")\n",
        "response = model.generate_content([video_file, prompt],\n",
        "                                  request_options={\"timeout\": 600})\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VosrkvAyrx-v"
      },
      "source": [
        "## List files\n",
        "\n",
        "You can list all uploaded files and their URIs using `files.list_files()`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "O82e6E2Irzlj"
      },
      "outputs": [],
      "source": [
        "# List all files\n",
        "for file in genai.list_files():\n",
        "    print(f\"{file.display_name}, URI: {file.uri}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "diCy9BgjLqeS"
      },
      "source": [
        "## Delete files\n",
        "\n",
        "Files are automatically deleted after 2 days. You can also manually delete them using `files.delete()`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "YYyi5PrKLqeb"
      },
      "outputs": [],
      "source": [
        "genai.delete_file(video_file.name)\n",
        "print(f'Deleted file {video_file.uri}')"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "vision.ipynb",
      "toc_visible": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}


Another vision notebook:

{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lb5yiH5h8x3h"
      },
      "source": [
        "##### Copyright 2024 Google LLC."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "cellView": "form",
        "id": "906e07f6e562"
      },
      "outputs": [],
      "source": [
        "#@title Licensed under the Apache License, Version 2.0 (the \"License\");\n",
        "# you may not use this file except in compliance with the License.\n",
        "# You may obtain a copy of the License at\n",
        "#\n",
        "# https://www.apache.org/licenses/LICENSE-2.0\n",
        "#\n",
        "# Unless required by applicable law or agreed to in writing, software\n",
        "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
        "# See the License for the specific language governing permissions and\n",
        "# limitations under the License."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WMGdicu8PVD9"
      },
      "source": [
        "# Video understanding with Gemini 2.0"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DR4Ti6Q0QKIl"
      },
      "source": [
        "<table align=\"left\">\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Video_understanding.ipynb\"><img src=\"https://github.com/google-gemini/cookbook/blob/main/images/colab_logo_32px.png?raw=1\" />Run in Google Colab</a>\n",
        "  </td>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3w14yjWnPVD-"
      },
      "source": [
        "Gemini has from the begining been a multimodal model, capable of analyzing all sorts of medias using its [long context window](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/).\n",
        "\n",
        "[Gemini 2.0](https://ai.google.dev/gemini-api/docs/models/gemini-v2) bring video analysis to a whole new level as illustrated in [this video](https://www.youtube.com/watch?v=Mot-JEU26GQ):\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "cellView": "form",
        "id": "CumMaR-sts53"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<iframe width=\"560\" height=\"315\" src=\"https://www.youtube.com/embed/Mot-JEU26GQ?si=pcb7-_MZTSi_1Zkw\" title=\"YouTube video player\" frameborder=\"0\" allow=\"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share\" referrerpolicy=\"strict-origin-when-cross-origin\" allowfullscreen></iframe>\n"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "#@title Building with Gemini 2.0: Video understanding\n",
        "%%html\n",
        "<iframe width=\"560\" height=\"315\" src=\"https://www.youtube.com/embed/Mot-JEU26GQ?si=pcb7-_MZTSi_1Zkw\" title=\"YouTube video player\" frameborder=\"0\" allow=\"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share\" referrerpolicy=\"strict-origin-when-cross-origin\" allowfullscreen></iframe>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jexx9acnuDsA"
      },
      "source": [
        "This notebook will show you how to easily use Gemini to perform the same kind of video analysis. Each of them has different prompts that you can select using the dropdown, also feel free to experiment with your own.\n",
        "\n",
        "You can also check the [live demo](https://aistudio.google.com/starter-apps/video) and try it on your own videos on [AI Studio](https://aistudio.google.com/starter-apps/video)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "R0HWzIEAQYqz"
      },
      "source": [
        "## Setup\n",
        "\n",
        "This section install the SDK, set it up using your [API key](../quickstarts/Authentication.ipynb), imports the relevant libs, downloads the sample videos and upload them to Gemini.\n",
        "\n",
        "Expand the section if you are curious, but you can also just run it (it should take a couple of minutes since there are large files) and go straight to the examples."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UzBKAaL4QYq0"
      },
      "source": [
        "### Install SDK\n",
        "\n",
        "The new **[Google Gen AI SDK](https://ai.google.dev/gemini-api/docs/sdks)** provides programmatic access to Gemini 2.0 (and previous models) using both the [Google AI for Developers](https://ai.google.dev/gemini-api/docs) and [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview) APIs. With a few exceptions, code that runs on one platform will run on both. This means that you can prototype an application using the Developer API and then migrate the application to Vertex AI without rewriting your code.\n",
        "\n",
        "More details about this new SDK on the [documentation](https://ai.google.dev/gemini-api/docs/sdks) or in the [Getting started](../gemini-2/get_started.ipynb) notebook."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "IbKkL5ksQYq1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/129.4 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━\u001b[0m \u001b[32m122.9/129.4 kB\u001b[0m \u001b[31m16.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m129.4/129.4 kB\u001b[0m \u001b[31m1.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ],
      "source": [
        "%pip install -U -q 'google-genai'"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aDUGen_kQYq2"
      },
      "source": [
        "### Setup your API key\n",
        "\n",
        "To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "0H_lRdlrQYq3"
      },
      "outputs": [],
      "source": [
        "from google.colab import userdata\n",
        "\n",
        "GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_3Lez1vBQYq3"
      },
      "source": [
        "### Initialize SDK client\n",
        "\n",
        "With the new SDK you now only need to initialize a client with you API key (or OAuth if using [Vertex AI](https://cloud.google.com/vertex-ai)). The model is now set in each call."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "X3CAp9YrQYq4"
      },
      "outputs": [],
      "source": [
        "from google import genai\n",
        "from google.genai import types\n",
        "\n",
        "client = genai.Client(api_key=GOOGLE_API_KEY)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ITgsQyaXQYq4"
      },
      "source": [
        "### Select the Gemini 2.0 model\n",
        "\n",
        "Video understanding works best Gemini 2.0 Flash model. You can also select former models to compare their behavior but it is recommended to use the 2.0 one.\n",
        "\n",
        "For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "IO7IoqbrQYq5"
      },
      "outputs": [],
      "source": [
        "model_name = \"gemini-2.0-flash\" # @param [\"gemini-1.5-flash-latest\",\"gemini-2.0-flash-lite-preview-02-05\",\"gemini-2.0-flash\",\"gemini-2.0-pro-exp-02-05\"] {\"allow-input\":true}"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5sgbw9krwPmo"
      },
      "source": [
        "### System instructions\n",
        "\n",
        " With the new SDK, the `system_instructions` and the `model` parameters must be passed in all `generate_content` calls, so let's save them to not have to type them all the time."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "6GBdoMUVwP6j"
      },
      "outputs": [],
      "source": [
        "system_instructions = \"\"\"\n",
        "    When given a video and a query, call the relevant function only once with the appropriate timecodes and text for the video\n",
        "  \"\"\"\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nVfyTD0zwv8p"
      },
      "source": [
        "These system instructions make sure the TODO: add explanation"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rv8ULT0lvJ47"
      },
      "source": [
        "### Get sample videos"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "fMcwUw48vL1N"
      },
      "outputs": [],
      "source": [
        "# Load sample images\n",
        "!wget https://storage.googleapis.com/generativeai-downloads/videos/Pottery.mp4 -O Pottery.mp4 -q\n",
        "!wget https://storage.googleapis.com/generativeai-downloads/videos/Jukin_Trailcam_Videounderstanding.mp4 -O Trailcam.mp4 -q\n",
        "!wget https://storage.googleapis.com/generativeai-downloads/videos/post_its.mp4 -O Post_its.mp4 -q\n",
        "!wget https://storage.googleapis.com/generativeai-downloads/videos/user_study.mp4 -O User_study.mp4 -q"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y4YMNQulz_yY"
      },
      "source": [
        "### Upload the videos\n",
        "\n",
        "Upload all the videos using the File API. You can find modre details about how to use it in the [Get Started](../gemini-2/get_started.ipynb#scrollTo=KdUjkIQP-G_i) notebook.\n",
        "\n",
        "This can take a couple of minutes as the videos will need to be processed and tokenized."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "LUUMJ4kE0OZS"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Waiting for video to be processed.\n",
            "Video processing complete: https://generativelanguage.googleapis.com/v1beta/files/o2b0tl8vxwjm\n",
            "Waiting for video to be processed.\n",
            "Waiting for video to be processed.\n",
            "Video processing complete: https://generativelanguage.googleapis.com/v1beta/files/qn0laf2skspy\n",
            "Waiting for video to be processed.\n",
            "Video processing complete: https://generativelanguage.googleapis.com/v1beta/files/3p82eu8fbz9p\n",
            "Waiting for video to be processed.\n",
            "Video processing complete: https://generativelanguage.googleapis.com/v1beta/files/lfvp1500v6sc\n"
          ]
        }
      ],
      "source": [
        "import time\n",
        "\n",
        "def upload_video(video_file_name):\n",
        "  video_file = client.files.upload(file=video_file_name)\n",
        "\n",
        "  while video_file.state == \"PROCESSING\":\n",
        "      print('Waiting for video to be processed.')\n",
        "      time.sleep(10)\n",
        "      video_file = client.files.get(name=video_file.name)\n",
        "\n",
        "  if video_file.state == \"FAILED\":\n",
        "    raise ValueError(video_file.state)\n",
        "  print(f'Video processing complete: ' + video_file.uri)\n",
        "\n",
        "  return video_file\n",
        "\n",
        "pottery_video = upload_video('Pottery.mp4')\n",
        "trailcam_video = upload_video('Trailcam.mp4')\n",
        "post_its_video = upload_video('Post_its.mp4')\n",
        "user_study_video = upload_video('User_study.mp4')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IF5tDbb-Q0oc"
      },
      "source": [
        "### Imports"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "B0Z9QzC3Q2wX"
      },
      "outputs": [],
      "source": [
        "import json\n",
        "from PIL import Image\n",
        "from IPython.display import display, Markdown, HTML"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GAa7sCD7tuMW"
      },
      "source": [
        "# Search within videos\n",
        "\n",
        "First, try using the model to search within your videos and describe all the animal sightings in the trailcam video.\n",
        "\n",
        "<video controls width=\"500\"><source src=\"https://storage.googleapis.com/generativeai-downloads/videos/Jukin_Trailcam_Videounderstanding.mp4\" type=\"video/mp4\"></video>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "PZw41-lsKKMf"
      },
      "outputs": [
        {
          "data": {
            "text/markdown": "Okay, here are the captions for each scene in the video, along with spoken text:\n\n```json\n[\n  {\n    \"00:00\": \"Close-up of a wild animal's fur, then the camera moves to the right, showing two gray foxes foraging among rocks and leaves in a forest.\"\n  },\n  {\n    \"00:16\": \"Black and white view of a mountain lion sniffing the ground in a forest setting.\"\n  },\n   {\n    \"00:35\": \"Black and white night vision shot of two foxes digging in the dirt by a tree. They briefly fight, and then one runs off.\"\n  },\n  {\n    \"01:05\": \"Night vision shot of a mountain lion, followed by a young mountain lion standing together by some rocks.\"\n  },\n  {\n   \"01:28\": \"Night vision shot of a bobcat standing and looking around in a forest.\"\n  },\n {\n    \"01:51\": \"A brown bear emerges from the forest and moves toward the trail camera.\"\n  },\n   {\n    \"01:57\": \"Black and white night vision shot of a mountain lion walking by a tree.\"\n  },\n  {\n    \"02:05\": \"Close-up of the side of a bear that is next to the trail camera.\"\n  },\n  {\n    \"02:23\": \"Night vision shot of a fox on a hillside, with a city visible in the distance.\"\n  },\n  {\n    \"02:35\": \"Black and white night vision shot of a bear in a forest setting.\"\n  },\n  {\n    \"02:51\": \"Night vision shot of a mountain lion behind a tree. It emerges and then runs off.\"\n  },\n {\n    \"03:05\": \"Two young brown bears are eating from the ground in a forest. A third, blacker bear is briefly visible in the background and then moves towards the camera and starts to eat from the ground. A plane flies overhead.\"\n  },\n {\n    \"04:22\": \"Night vision shot of a bobcat looking around and then walking off into the forest.\"\n  },\n  {\n   \"04:57\": \"Night vision shot of a mountain lion eating from the ground in a forest setting.\"\n  }\n]\n```\n",
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "execution_count": 10,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "prompt = \"For each scene in this video, generate captions that describe the scene along with any spoken text placed in quotation marks. Place each caption into an object with the timecode of the caption in the video.\"  # @param [\"For each scene in this video, generate captions that describe the scene along with any spoken text placed in quotation marks. Place each caption into an object with the timecode of the caption in the video.\", \"Organize all scenes from this video in a table, along with timecode, a short description, a list of objects visible in the scene (with representative emojis) and an estimation of the level of excitement on a scale of 1 to 10\"] {\"allow-input\":true}\n",
        "\n",
        "video = trailcam_video # @param [\"trailcam_video\", \"pottery_video\", \"post_its_video\", \"user_study_video\"] {\"type\":\"raw\",\"allow-input\":true}\n",
        "\n",
        "response = client.models.generate_content(\n",
        "    model=model_name,\n",
        "    contents=[\n",
        "        video,\n",
        "        prompt,\n",
        "    ]\n",
        ")\n",
        "\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nOQzKYGJKAnD"
      },
      "source": [
        "The prompt used is quite a generic one, but you can get even better results if you cutomize it to your needs (like asking specifically for foxes).\n",
        "\n",
        "The [live demo on AI Studio](https://aistudio.google.com/starter-apps/video) shows how you can postprocess this output to jump directly to the the specific part of the video by clicking on the timecodes. If you are interested, you can check the [code of that demo on Github](https://github.com/google-gemini/starter-applets/tree/main/video)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Wog32E7CKnT6"
      },
      "source": [
        "# Extract and organize text\n",
        "\n",
        "Gemini can also read what's in the video and extract it in an organized way. You can even use Gemini reasoning capabilities to generate new ideas for you.\n",
        "\n",
        "<video controls width=\"400\"><source src=\"https://storage.googleapis.com/generativeai-downloads/videos/post_its.mp4\" type=\"video/mp4\"></video>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "baNCeA3GKrfu"
      },
      "outputs": [
        {
          "data": {
            "text/markdown": "Absolutely! Here's a transcription of the sticky notes, organized into a table, along with some additional name ideas:\n\n| Category           | Name Ideas                 |\n| ------------------ | ---------------------------- |\n| Constellations     | Leo Minor, Canis Major, Orion's Belt,  Lyra,   |\n|                    | Draco, Delphinus, Centaurus,  Serpens,       |\n|                    |                               |\n| Celestial Bodies   | Lunar Eclipse,   Stellar Nexus,            |\n|                    | Supernova Echo, Comet's Tail, Galactic Core, Titan |\n|                    | Phoenix   Medusa              |\n| Mythology          | Astral Forge, Prometheus Rising,  Chimera Dream,      |\n|                    | Persius Shield,   Zephyr,   Odin Athena Hera  |\n|                    | Aether, Cerberus                |\n| Science/Math       | Convergence, Chaos Field, Euler's Path,   |\n|                    | Equilibrium,   Chaos Theory,       |\n|                    | Golden Ratio, Infinity Loop,  Bayes Theorem |\n|                    | Riemann's Hypothesis, Taylor Series,   |\n|                    | Stokes Theorem, Fractal             |\n| Other              | Andromeda's Reach, Lynx,  Vector, |\n|                    | Athena's Eye, Sagitta Pandora's Box, Celestial Drift|\n|                   |  Symmetry                     |\n\n**Additional Project Name Ideas:**\n\n| Category           | Name Ideas                 |\n| ------------------ | ---------------------------- |\n| Constellations     | Cygnus, Hercules, Virgo, Aries, Taurus, Gemini, Cancer, Scorpius, Capricornus, Aquarius, Pisces, Ursa Major, Ursa Minor, Cassiopeia |\n| Celestial Bodies   | Aurora Borealis, Cosmic Ray, Nova Spark, Black Hole Horizon, Quantum Star, Nebula Veil, Asteroid Field, Pulsar Echo, Void Walker |\n| Mythology          | Hephaestus, Elysium, Mimir's Well, Valhalla,  Ouroboros, Gaea, Tartarus,  The Gorgon, Olympus Peak |\n| Science/Math       | Heisenberg Certainty, Boltzmann Brain, Fibonacci Sequence, Cantor Set, Gödel's Theorem, Turing Machine, Markov Chain, String Theory |\n| Other              | Temporal Rift, Entropy's End, Singularity Surge,  Quicksilver Dream, Labyrinthine Echo, Shadow Cascade, Elysian Fields |\n",
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "execution_count": 20,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "prompt = \"Transcribe the sticky notes, organize them and put it in a table. Can you come up with a few more ideas?\" # @param [\"Transcribe the sticky notes, organize them and put it in a table. Can you come up with a few more ideas?\", \"Which of those names who fit an AI product that can resolve complex questions using its thinking abilities?\"] {\"allow-input\":true}\n",
        "\n",
        "video = post_its_video # @param [\"trailcam_video\", \"pottery_video\", \"post_its_video\", \"user_study_video\"] {\"type\":\"raw\",\"allow-input\":true}\n",
        "\n",
        "response = client.models.generate_content(\n",
        "    model=model_name,\n",
        "    contents=[\n",
        "        video,\n",
        "        prompt,\n",
        "    ]\n",
        ")\n",
        "\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DjKIsLDMTNk1"
      },
      "source": [
        "# Structure information\n",
        "\n",
        "Gemini 2.0 is not only able to read text but also to reason and structure about real world objects. Like in this video about a display of ceramics with handwritten prices and notes.\n",
        "\n",
        "<video controls width=\"500\"><source src=\"https://storage.googleapis.com/generativeai-downloads/videos/Pottery.mp4\" type=\"video/mp4\"></video>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "bqzqedMFT5Wp"
      },
      "outputs": [
        {
          "data": {
            "text/markdown": "Sure! Here's a table of the items and notes based on the image:\n\n| Item          | Description             | Dimensions (approximate)  | Price   | Notes                      |\n|---------------|-------------------------|------------------------|---------|-----------------------------|\n| Tumblers      | #5 Artichoke double dip | 4\"h x 3\"d                 | \\$20    | -ISH                       |\n| Small Bowls   | -                      | 3.5\"h x 6.5\"d                 | \\$35    | -                        |\n| Medium Bowls  | -                       | 4\"h x 7\"d                 | \\$40    | -                         |\n| Glaze Tile   | #6 Gemini double dip   | -        | -    | Slow Cool                 |",
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "execution_count": 21,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "prompt = \"Give me a table of my items and notes\" # @param [\"Give me a table of my items and notes\", \"Help me come up with a selling pitch for my potteries\"] {\"allow-input\":true}\n",
        "\n",
        "video = pottery_video # @param [\"trailcam_video\", \"pottery_video\", \"post_its_video\", \"user_study_video\"] {\"type\":\"raw\",\"allow-input\":true}\n",
        "\n",
        "response = client.models.generate_content(\n",
        "    model=model_name,\n",
        "    contents=[\n",
        "        video,\n",
        "        prompt,\n",
        "    ],\n",
        "    config = types.GenerateContentConfig(\n",
        "        system_instruction=\"Don't forget to escape the dollar signs\",\n",
        "    )\n",
        ")\n",
        "\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Zsh6i-Z6VHNK"
      },
      "source": [
        "As you can see, Gemini is able to grasp to with item corresponds each note, including the last one."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uIfsFC0pVUTD"
      },
      "source": [
        "# Analyze screen recordings for key moments\n",
        "\n",
        "You can also use the model to analyze screen recordings. Let's say you're doing user studies on how people use your product, so you end up with lots of screen recordings, like this one, that you have to manually comb through.\n",
        "With just one prompt, the model can describe all the actions in your video.\n",
        "\n",
        "<video controls width=\"400\"><source src=\"https://storage.googleapis.com/generativeai-downloads/videos/user_study.mp4\" type=\"video/mp4\"></video>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "wrMHZ0MxW75y"
      },
      "outputs": [
        {
          "data": {
            "text/markdown": "Here is a summary of the video:\n\nThe video shows a person using \"My Garden App\" to add plants to their cart. They start by liking and adding a Fern (0:10-0:14) and Cactus to their cart. (0:15-0:17) They then select a Hibiscus (0:22-0:25), and an Orchid. (0:44-0:45) Finally, they check their cart and profile to confirm the plants they have liked and added to the cart. (0:30-0:35)\n",
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "execution_count": 23,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "prompt = \"Generate a paragraph that summarizes this video. Keep it to 3 to 5 sentences with corresponding timecodes.\" # @param [\"Generate a paragraph that summarizes this video. Keep it to 3 to 5 sentences with corresponding timecodes.\", \"Choose 5 key shots from this video and put them in a table with the timecode, text description of 10 words or less, and a list of objects visible in the scene (with representative emojis).\", \"Generate bullet points for the video. Place each bullet point into an object with the timecode of the bullet point in the video.\"] {\"allow-input\":true}\n",
        "\n",
        "video = user_study_video # @param [\"trailcam_video\", \"pottery_video\", \"post_its_video\", \"user_study_video\"] {\"type\":\"raw\",\"allow-input\":true}\n",
        "\n",
        "response = client.models.generate_content(\n",
        "    model=model_name,\n",
        "    contents=[\n",
        "        video,\n",
        "        prompt,\n",
        "    ]\n",
        ")\n",
        "\n",
        "Markdown(response.text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cizoUEdIYLd0"
      },
      "source": [
        "Once again, you can check the  [live demo on AI Studio](https://aistudio.google.com/starter-apps/video) shows an example on how to postprocess this output. Check the [code of that demo](https://github.com/google-gemini/starter-applets/tree/main/video) for more details."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lND4jB6MrsSk"
      },
      "source": [
        "# Next Steps\n",
        "\n",
        "Try with you own videos using the [AI Studio's live demo](https://aistudio.google.com/starter-apps/video) or play with the examples from this notebook (in case you haven't seen, there are other prompts you can try in the dropdowns).\n",
        "\n",
        "For more examples of the Gemini 2.0 capabilities, check the [Gemini 2.0 folder of the cookbook](https://github.com/google-gemini/cookbook/tree/main/gemini-2/). You'll learn how to use the [Live API](../quickstarts/Get_started_LiveAPI.ipynb), juggle with [multiple tools](../quickstarts/Get_started_LiveAPI_tools.ipynb) or use Gemini 2.0 [spatial understanding](../quickstarts/Spatial_understanding.ipynb) abilities.\n",
        "\n",
        "The [examples](https://github.com/google-gemini/cookbook/tree/main/examples/) folder from the cookbook is also full of nice code samples illustrating creative ways to use Gemini multimodal capabilities and long-context."
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "Video_understanding.ipynb",
      "toc_visible": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}


Search capabilities:

Grounding with Google Search

Python Node.js REST

The Grounding with Google Search feature in the Gemini API and AI Studio can be used to improve the accuracy and recency of responses from the model. In addition to more factual responses, when Grounding with Google Search is enabled, the Gemini API returns grounding sources (in-line supporting links) and Google Search Suggestions along with the response content. The Search Suggestions point users to the search results corresponding to the grounded response.

This guide will help you get started with Grounding with Google Search using one of the Gemini API SDKs or the REST API.

Configure a model to use Google Search
To configure a model to use Grounding with Google Search, pass in the appropriate tool.

Getting started

from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents="Who won Wimbledon this year?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            google_search=types.GoogleSearchRetrieval
        )]
    )
)
print(response)
Dynamic threshold
The dynamic_threshold settings let you control the retrieval behavior, giving you additional control over when Grounding with Google Search is used.


from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents="Who won Wimbledon this year?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            google_search=types.GoogleSearchRetrieval(
                dynamic_retrieval_config=types.DynamicRetrievalConfig(
                    dynamic_threshold=0.6))
        )]
    )
)
print(response)
Grounding with Google Search works with all available languages when doing text prompts. On the paid tier of the Gemini Developer API, 1,500 Grounding with Google Search queries per day are free, with additional queries billed at the standard $35 per 1,000 queries.

Why is Grounding with Google Search useful?
In generative AI, grounding refers to the process of connecting the model to verifiable sources of information. These sources might provide real-world workplace information or other specific context. Grounding helps with improving the accuracy, reliability, and usefulness of AI outputs.

Grounding is particularly important for prompts that require up-to-date information from the web. Using grounding, the model can access information beyond its knowledge cutoff date, get sources for the information, and answer questions that it couldn't have answered accurately otherwise.

Using Google AI Studio or the Gemini API, you can ground model output to Google Search. Grounding with Google Search provides the following benefits:

Allows model responses that are tethered to specific content.
Reduces model hallucinations, which are instances where the model generates content that isn't factual.
Anchors model responses to sources a user can click through and open.
Enhances the trustworthiness and applicability of the generated content.
When you use Grounding with Google Search, you're effectively connecting the model to reliable Search results from the internet. Since non-grounded model responses are based on learned patterns, you might not get factual responses to prompts about current events (for example, asking for a weather forecast or the final score of a recent football game). Since the internet provides access to new information, a grounded prompt can generate more up-to-date responses, with sources cited.

Here's an example comparing a non-grounded response and a grounded response generated using the API. (The responses were generated in October 2024.)

Ungrounded Gemini	Grounding with Google Search
Prompt: Who won the Super Bowl this year?
Response: The Kansas City Chiefs won Super Bowl LVII this year (2023).	Prompt: Who won the Super Bowl this year?
Response: The Kansas City Chiefs won Super Bowl LVIII this year, defeating the San Francisco 49ers in overtime with a score of 25 to 22.
In the ungrounded response, the model refers to the Kansas City Chiefs' 2023 Super Bowl win. In the grounded response, the model correctly references their more recent 2024 win.

The following image shows how a grounded response looks in AI Studio.

Grounded prompt and response in AI Studio

Google Search Suggestions
To use Grounding with Google Search, you have to display Google Search Suggestions, which are suggested queries included in the metadata of the grounded response. To learn more about the display requirements, see Use Google Search Suggestions.

Dynamic retrieval
Some queries are likely to benefit more from Grounding with Google Search than others. The dynamic retrieval feature gives you additional control over when to use Grounding with Google Search.

If the dynamic retrieval mode is unspecified, Grounding with Google Search is always triggered. If the mode is set to dynamic, the model decides when to use grounding based on a threshold that you can configure. The threshold is a floating-point value in the range [0,1] and defaults to 0.3. If the threshold value is 0, the response is always grounded with Google Search; if it's 1, it never is.

How dynamic retrieval works
You can use dynamic retrieval in your request to choose when to turn on Grounding with Google Search. This is useful when the prompt doesn't require an answer grounded in Google Search and the model can provide an answer based on its own knowledge without grounding. This helps you manage latency, quality, and cost more effectively.

Before you invoke the dynamic retrieval configuration in your request, understand the following terminology:

Prediction score: When you request a grounded answer, Gemini assigns a prediction score to the prompt. The prediction score is a floating point value in the range [0,1]. Its value depends on whether the prompt can benefit from grounding the answer with the most up-to-date information from Google Search. Thus, if a prompt requires an answer grounded in the most recent facts on the web, it has a higher prediction score. A prompt for which a model-generated answer is sufficient has a lower prediction score.

Here are examples of some prompts and their prediction scores.

Note: The prediction scores are assigned by Gemini and can vary over time depending on several factors.
Prompt	Prediction score	Comment
"Write a poem about peonies"	0.13	The model can rely on its knowledge and the answer doesn't need grounding.
"Suggest a toy for a 2yo child"	0.36	The model can rely on its knowledge and the answer doesn't need grounding.
"Can you give a recipe for an asian-inspired guacamole?"	0.55	Google Search can give a grounded answer, but grounding isn't strictly required; the model knowledge might be sufficient.
"What's Agent Builder? How is grounding billed in Agent Builder?"	0.72	Requires Google Search to generate a well-grounded answer.
"Who won the latest F1 grand prix?"	0.97	Requires Google Search to generate a well-grounded answer.
Threshold: In your API request, you can specify a dynamic retrieval configuration with a threshold. The threshold is a floating point value in the range [0,1] and defaults to 0.3. If the threshold value is zero, the response is always grounded with Google Search. For all other values of threshold, the following is applicable:

If the prediction score is greater than or equal to the threshold, the answer is grounded with Google Search. A lower threshold implies that more prompts have responses that are generated using Grounding with Google Search.
If the prediction score is less than the threshold, the model might still generate the answer, but it isn't grounded with Google Search.
To learn how to set the dynamic retrieval threshold using an SDK or the REST API, see the appropriate code example.

If you're using AI Studio, you can set the dynamic retrieval threshold by clicking Edit grounding.

Select grounding threshold in AI Studio

To find a good threshold that suits your business needs, you can create a representative set of queries that you expect to encounter. Then you can sort the queries according to the prediction score in the response and select a good threshold for your use case.

Search as a tool
Starting with Gemini 2.0, Google Search is available as a tool. This means that the model can decide when to use Google Search. The following example shows how to configure Search as a tool.


from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

client = genai.Client()
model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

response = client.models.generate_content(
    model=model_id,
    contents="When is the next total solar eclipse in the United States?",
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )
)

for each in response.candidates[0].content.parts:
    print(each.text)
# Example response:
# The next total solar eclipse visible in the contiguous United States will be on ...

# To get grounding metadata as web content.
print(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)
The Search-as-a-tool functionality also enables multi-turn searches and multi-tool queries (for example, combining Grounding with Google Search and code execution).

Search as a tool enables complex prompts and workflows that require planning, reasoning, and thinking:

Grounding to enhance factuality and recency and provide more accurate answers
Retrieving artifacts from the web to do further analysis on
Finding relevant images, videos, or other media to assist in multimodal reasoning or generation tasks
Coding, technical troubleshooting, and other specialized tasks
Finding region-specific information or assisting in translating content accurately
Finding relevant websites for further browsing
You can get started by trying the Search tool notebook.

A grounded response
If your prompt successfully grounds to Google Search, the response will include groundingMetadata. A grounded response might look something like this (parts of the response have been omitted for brevity):


{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "Carlos Alcaraz won the Gentlemen's Singles title at the 2024 Wimbledon Championships. He defeated Novak Djokovic in the final, winning his second consecutive Wimbledon title and fourth Grand Slam title overall. \n"
          }
        ],
        "role": "model"
      },
      ...
      "groundingMetadata": {
        "searchEntryPoint": {
          "renderedContent": "\u003cstyle\u003e\n.container {\n  align-items: center;\n  border-radius: 8px;\n  display: flex;\n  font-family: Google Sans, Roboto, sans-serif;\n  font-size: 14px;\n  line-height: 20px;\n  padding: 8px 12px;\n}\n.chip {\n  display: inline-block;\n  border: solid 1px;\n  border-radius: 16px;\n  min-width: 14px;\n  padding: 5px 16px;\n  text-align: center;\n  user-select: none;\n  margin: 0 8px;\n  -webkit-tap-highlight-color: transparent;\n}\n.carousel {\n  overflow: auto;\n  scrollbar-width: none;\n  white-space: nowrap;\n  margin-right: -12px;\n}\n.headline {\n  display: flex;\n  margin-right: 4px;\n}\n.gradient-container {\n  position: relative;\n}\n.gradient {\n  position: absolute;\n  transform: translate(3px, -9px);\n  height: 36px;\n  width: 9px;\n}\n@media (prefers-color-scheme: light) {\n  .container {\n    background-color: #fafafa;\n    box-shadow: 0 0 0 1px #0000000f;\n  }\n  .headline-label {\n    color: #1f1f1f;\n  }\n  .chip {\n    background-color: #ffffff;\n    border-color: #d2d2d2;\n    color: #5e5e5e;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #f2f2f2;\n  }\n  .chip:focus {\n    background-color: #f2f2f2;\n  }\n  .chip:active {\n    background-color: #d8d8d8;\n    border-color: #b6b6b6;\n  }\n  .logo-dark {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);\n  }\n}\n@media (prefers-color-scheme: dark) {\n  .container {\n    background-color: #1f1f1f;\n    box-shadow: 0 0 0 1px #ffffff26;\n  }\n  .headline-label {\n    color: #fff;\n  }\n  .chip {\n    background-color: #2c2c2c;\n    border-color: #3c4043;\n    color: #fff;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #353536;\n  }\n  .chip:focus {\n    background-color: #353536;\n  }\n  .chip:active {\n    background-color: #464849;\n    border-color: #53575b;\n  }\n  .logo-light {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);\n  }\n}\n\u003c/style\u003e\n\u003cdiv class=\"container\"\u003e\n  \u003cdiv class=\"headline\"\u003e\n    \u003csvg class=\"logo-light\" width=\"18\" height=\"18\" viewBox=\"9 9 35 35\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"\u003e\n      \u003cpath fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z\" fill=\"#4285F4\"/\u003e\n      \u003cpath fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z\" fill=\"#34A853\"/\u003e\n      \u003cpath fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z\" fill=\"#FBBC05\"/\u003e\n      \u003cpath fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z\" fill=\"#EA4335\"/\u003e\n    \u003c/svg\u003e\n    \u003csvg class=\"logo-dark\" width=\"18\" height=\"18\" viewBox=\"0 0 48 48\" xmlns=\"http://www.w3.org/2000/svg\"\u003e\n      \u003ccircle cx=\"24\" cy=\"23\" fill=\"#FFF\" r=\"22\"/\u003e\n      \u003cpath d=\"M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z\" fill=\"#4285F4\"/\u003e\n      \u003cpath d=\"M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z\" fill=\"#34A853\"/\u003e\n      \u003cpath d=\"M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z\" fill=\"#FBBC05\"/\u003e\n      \u003cpath d=\"M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z\" fill=\"#EA4335\"/\u003e\n    \u003c/svg\u003e\n    \u003cdiv class=\"gradient-container\"\u003e\u003cdiv class=\"gradient\"\u003e\u003c/div\u003e\u003c/div\u003e\n  \u003c/div\u003e\n  \u003cdiv class=\"carousel\"\u003e\n    \u003ca class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4x8Epe-gzpwRBvp7o3RZh2m1ygq1EHktn0OWCtvTXjad4bb1zSuqfJd6OEuZZ9_SXZ_P2SvCpJM7NaFfQfiZs6064MeqXego0vSbV9LlAZoxTdbxWK1hFeqTG6kA13YJf7Fbu1SqBYM0cFM4zo0G_sD9NKYWcOCQMvDLDEJFhjrC9DM_QobBIAMq-gWN95G5tvt6_z6EuPN8QY=\"\u003ewho won wimbledon 2024\u003c/a\u003e\n  \u003c/div\u003e\n\u003c/div\u003e\n"
        },
        "groundingChunks": [
          {
            "web": {
              "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4whET1ta3sDETZvcicd8FeNe4z0VuduVsxrT677KQRp2rYghXI0VpfYbIMVI3THcTuMwggRCbFXS_wVvW0UmGzMe9h2fyrkvsnQPJyikJasNIbjJLPX0StM4Bd694-ZVle56MmRA4YiUvwSqad1w6O2opmWnw==",
              "title": "wikipedia.org"
            }
          },
          {
            "web": {
              "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4wR1M-9-yMPUr_KdHlnoAmQ8ZX90DtQ_vDYTjtP2oR5RH4tRP04uqKPLmesvo64BBkPeYLC2EpVDxv9ngO3S1fs2xh-e78fY4m0GAtgNlahUkm_tBm_sih5kFPc7ill9u2uwesNGUkwrQlmP2mfWNU5lMMr23HGktr6t0sV0QYlzQq7odVoBxYWlQ_sqWFH",
              "title": "wikipedia.org"
            }
          },
          {
            "web": {
              "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4wsDmROzbP-tmt8GdwCW_pqISTZ4IRbBuoaMyaHfcQg8WW-yKRQQvMDTPAuLxJh-8_U8_iw_6JKFbQ8M9oVYtaFdWFK4gOtL4RrC9Jyqc5BNpuxp6uLEKgL5-9TggtNvO97PyCfziDFXPsxylwI1HcfQdrz3Jy7ZdOL4XM-S5rC0lF2S3VWW0IEAEtS7WX861meBYVjIuuF_mIr3spYPqWLhbAY2Spj-4_ba8DjRvmevIFUhRuESTKvBfmpxNSM",
              "title": "cbssports.com"
            }
          },
          {
            "web": {
              "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4yzjLkorHiUKjhOPkWaZ9b4cO-cLG-02vlEl6xTBjMUjyhK04qSIclAa7heR41JQ6AAVXmNdS3WDrLOV4Wli-iezyzW8QPQ4vgnmO_egdsuxhcGk3-Fp8-yfqNLvgXFwY5mPo6QRhvplOFv0_x9mAcka18QuAXtj0SPvJfZhUEgYLCtCrucDS5XFc5HmRBcG1tqFdKSE1ihnp8KLdaWMhrUQI21hHS9",
              "title": "jagranjosh.com"
            }
          },
          {
            "web": {
              "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AWhgh4y9L4oeNGWCatFz63b9PpP3ys-Wi_zwnkUT5ji9lY7gPUJQcsmmE87q88GSdZqzcx5nZG9usot5FYk2yK-FAGvCRE6JsUQJB_W11_kJU2HVV1BTPiZ4SAgm8XDFIxpCZXnXmEx5HUfRqQm_zav7CvS2qjA2x3__qLME6Jy7R5oza1C5_aqjQu422le9CaigThS5bvJoMo-ZGcXdBUCj2CqoXNVjMA==",
              "title": "apnews.com"
            }
          }
        ],
        "groundingSupports": [
          {
            "segment": {
              "endIndex": 85,
              "text": "Carlos Alcaraz won the Gentlemen's Singles title at the 2024 Wimbledon Championships."
            },
            "groundingChunkIndices": [
              0,
              1,
              2,
              3
            ],
            "confidenceScores": [
              0.97380733,
              0.97380733,
              0.97380733,
              0.97380733
            ]
          },
          {
            "segment": {
              "startIndex": 86,
              "endIndex": 210,
              "text": "He defeated Novak Djokovic in the final, winning his second consecutive Wimbledon title and fourth Grand Slam title overall."
            },
            "groundingChunkIndices": [
              1,
              0,
              4
            ],
            "confidenceScores": [
              0.96145374,
              0.96145374,
              0.96145374
            ]
          }
        ],
        "webSearchQueries": [
          "who won wimbledon 2024"
        ]
      }
    }
  ],
  ...
}
If the response doesn't include groundingMetadata, this means the response wasn't successfully grounded. There are several reasons this could happen, including low source relevance or incomplete information within the model response.

When a grounded result is generated, the metadata contains URIs that redirect to the publishers of the content that was used to generate the grounded result. These URIs contain the vertexaisearch subdomain, as in this truncated example: https://vertexaisearch.cloud.google.com/grounding-api-redirect/.... The metadata also contains the publishers' domains. The provided URIs remain accessible for 30 days after the grounded result is generated.

Important: The provided URIs must be directly accessible by the end users and must not be queried programmatically through automated means. If automated access is detected, the grounded answer generation service might stop providing the redirection URIs.


Use Google Search Suggestions

To use Grounding with Google Search, you must enable Google Search Suggestions, which help users find search results corresponding to a grounded response.

Specifically, you need to display the search queries that are included in the grounded response's metadata. The response includes:

content: LLM generated response
webSearchQueries: The queries to be used for Google Search Suggestions
For example, in the following code snippet, Gemini responds to a search-grounded prompt which is asking about a type of tropical plant.


"predictions": [
  {
    "content": "Monstera is a type of vine that thrives in bright indirect light…",
    "groundingMetadata": {
      "webSearchQueries": ["What's a monstera?"],
    }
  }
]
You can take this output and display it by using Google Search Suggestions.

Requirements for Google Search Suggestions
Do:

Display the Search Suggestion exactly as provided without any modifications while complying with the Display Requirements.
Take users directly to the Google Search results page (SRP) when they interact with the Search Suggestion.
Don't:

Include any interstitial screens or additional steps between the user's tap and the display of the SRP.
Display any other search results or suggestions alongside the Search Suggestion or associated grounded LLM response.
Display requirements
Display the Search Suggestion exactly as provided and don't make any modifications to colors, fonts, or appearance. Ensure the Search Suggestion renders as specified in the following mocks, including for light and dark mode: mock Search Suggestion display

Whenever a grounded response is shown, its corresponding Google Search Suggestion should remain visible.

Branding: You must strictly follow Google's Guidelines for Third Party Use of Google Brand Features.

Google Search Suggestions should be at minimum the full width of the grounded response.

Behavior on tap
When a user taps the chip, they are taken directly to a Google Search results page (SRP) for the search term displayed in the chip. The SRP can open either within your in-app browser or in a separate browser app. It's important to not minimize, remove, or obstruct the SRP's display in any way. The following animated mockup illustrates the tap-to-SRP interaction.

user navigating to search results page

Code to implement a Google Search Suggestion
When you use the API to ground a response to search, the model response provides compliant HTML and CSS styling in the renderedContent field which you implement to display Search Suggestions in your application. To see an example of the API response, see the response section in Grounding with Google Search.

Note: The provided HTML and CSS provided in the API response automatically adapts to the user's device settings, displaying in either light or dark mode based on the user's preference indicated by @media(prefers-color-scheme).



Search notebook:

gle Colab
In this notebook you will learn how to use the new Google Search tool available in Gemini 2.0, using both the unary API and the Multimodal Live API. Check out the docs to learn more about using Search as a tool.

Note that the previous version of this guide using Gemini models priori to 2.0 and the legacy SDK can still be found here.

Set up the SDK
This guide uses the google-genai Python SDK to connect to the Gemini 2.0 models.

Install SDK
The new Google Gen AI SDK provides programmatic access to Gemini 2 (and previous models) using both the Google AI for Developers and Vertex AI APIs. With a few exceptions, code that runs on one platform will run on both. This means that you can prototype an application using the Developer API and then migrate the application to Vertex AI without rewriting your code.

More details about this new SDK on the documentation or in the Getting started notebook.


!pip install -U -q google-genai
     
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/110.9 kB ? eta -:--:--
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━ 102.4/110.9 kB 7.1 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 110.9/110.9 kB 2.6 MB/s eta 0:00:00
Set up your API key
To run the following cell, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the Authentication quickstart for an example.


import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
     
Initialize SDK client
The client will pick up your API key from the environment variable. To use the live API you need to set the client version to v1alpha and use the Gemini 2.0 model.


from google import genai

client = genai.Client(http_options={'api_version': 'v1alpha'})

MODEL = 'gemini-2.0-flash-exp'
     
Use search in chat
Start by defining a helper function that you will use to display each part of the returned response.


# @title Define some helpers (run this cell)
import json

from IPython.display import display, HTML, Markdown


def show_json(obj):
  print(json.dumps(obj.model_dump(exclude_none=True), indent=2))

def show_parts(r):
  parts = r.candidates[0].content.parts
  if parts is None:
    finish_reason = r.candidates[0].finish_reason
    print(f'{finish_reason=}')
    return
  for part in r.candidates[0].content.parts:
    if part.text:
      display(Markdown(part.text))
    elif part.executable_code:
      display(Markdown(f'```python\n{part.executable_code.code}\n```'))
    else:
      show_json(part)

  grounding_metadata = r.candidates[0].grounding_metadata
  if grounding_metadata and grounding_metadata.search_entry_point:
    display(HTML(grounding_metadata.search_entry_point.rendered_content))

     
First try a query that needs realtime information, so you can see how the model performs without Google Search.


chat = client.chats.create(model=MODEL)

r = chat.send_message('Who won the most recent Australia vs Chinese Taipei games?')
show_parts(r)
     
To figure out who won the most recent game between Australia and Chinese Taipei, I need to know which sport you're referring to, as they compete in many different sports.

Please specify the sport you're interested in.

For example, are you asking about:

Soccer (Football)?
Basketball?
Baseball?
Other sport?
Once you tell me the sport, I can give you the correct information.

Now set up a new chat session that uses the google_search tool. The show_parts helper will display the text output as well as any Google Search queries used in the results.


search_tool = {'google_search': {}}
soccer_chat = client.chats.create(model=MODEL, config={'tools': [search_tool]})

r = soccer_chat.send_message('Who won the most recent Australia vs Chinese Taipei games?')
show_parts(r)
     
The most recent games between Australia and Chinese Taipei were in a friendly series in December 2024. The Australian women's national team, known as the Matildas, played against Chinese Taipei twice, winning both matches.

December 7, 2024: Australia won 6-0 against Chinese Taipei in Geelong. The goals were scored by Leah Davidson, Tameka Yallop, Emily Gielnik, Michelle Heyman, Tash Prior and Sharn Freier.
The first match of the series was also won by Australia, although the specific score is not detailed in the provided context.
Prior to these two matches, Australia and Chinese Taipei have played a total of 2 matches since 2012. Australia won both of those matches as well. In addition, the Australian women's team has played against Chinese Taipei's women's team six times since 2008, winning all six games.

most recent Australia vs Chinese Taipei games Australia vs Chinese Taipei results
As you are using a chat session, you can ask the model follow-up questions too.


r = soccer_chat.send_message('Who scored the goals?')
show_parts(r)
     
Here are the goal scorers for the two recent matches between Australia and Chinese Taipei in December 2024:

December 7, 2024 (Australia 6 - Chinese Taipei 0):

Leah Davidson scored her first international goal with a header in the 6th minute, assisted by Emily van Egmond.
Tameka Yallop scored in the 11th minute, with a left-footed shot.
Emily Gielnik scored with a header in the 40th minute, assisted by Tameka Yallop.
Michelle Heyman scored in the 56th minute, assisted by Remy Siemsen.
Tash Prior scored in the 73rd minute, assisted by Emily van Egmond.
Sharn Freier scored in the 78th minute, assisted by Winonah Heatley.
December 4, 2024 (Australia 3 - Chinese Taipei 1):

Tash Prior scored her first international goal in the 10th minute, assisted by Chloe Logarzo.
Sharn Freier scored her first international goal in the 12th minute after a shot from Emily Gielnik hit the crossbar and fell to her at the left post.
Bryleeh Henry scored her first international goal in the 78th minute, assisted by Tameka Yallop
Chen Jin-Wen scored for Chinese Taipei in the 34th minute.
In summary, the goal scorers across the two matches were: Leah Davidson, Tameka Yallop, Emily Gielnik, Michelle Heyman, Tash Prior, Sharn Freier, Bryleeh Henry for Australia, and Chen Jin-Wen for Chinese Taipei.

Australia vs Chinese Taipei soccer December 2024 goal scorers who scored goals Australia vs Chinese Taipei December 2024
Plot search results
In this example you can see how to use the Google Search tool with code generation in order to plot results.


movie_chat = client.chats.create(model=MODEL, config={'tools': [search_tool]})

r = movie_chat.send_message('Generate some Python code to plot the runtimes of the last 10 Denis Villeneuve movies.')
show_parts(r)
     
Okay, here's some Python code using matplotlib to plot the runtimes of Denis Villeneuve's last 10 movies.

import matplotlib.pyplot as plt

# Movie titles and runtimes in minutes
movies = [
    "Dune: Part Two", "Dune: Part One", "Blade Runner 2049", "Arrival", "Sicario",
    "Enemy", "Prisoners", "Incendies", "Polytechnique", "Maelström"
]
runtimes = [166, 155, 164, 116, 121, 91, 153, 131, 77, 87]


# Create the bar plot
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
plt.bar(movies, runtimes, color='skyblue')

# Add labels and title
plt.xlabel("Movies", fontsize = 12)
plt.ylabel("Runtime (minutes)", fontsize = 12)
plt.title("Denis Villeneuve Movie Runtimes (Last 10 Movies)", fontsize = 14)
plt.xticks(rotation=45, ha="right") # Rotate x-axis labels for better readability

# Add the exact runtime on top of each bar
for i, runtime in enumerate(runtimes):
    plt.text(i, runtime + 1, str(runtime), ha='center', va='bottom') # Adjust 'runtime + 1' if needed to place text better


# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Display the plot
plt.show()
Explanation:

Import matplotlib.pyplot: This line imports the necessary library for plotting.
Data:
movies: A list containing the titles of Denis Villeneuve's last 10 movies.
runtimes: A list containing the corresponding runtimes in minutes. The runtimes were gathered from the search results. The most recent movie being "Dune: Part Two" and the oldest movie from the last 10 being "Maelstrom".
Create Plot:
plt.figure(figsize=(12, 6)): Creates a figure and axes object, setting the figure size.
plt.bar(movies, runtimes, color='skyblue'): Creates a bar plot with movie titles on the x-axis and runtimes on the y-axis.
plt.xlabel(), plt.ylabel(), plt.title(): Adds labels and a title to the plot.
plt.xticks(rotation=45, ha="right"): Rotates the x-axis labels for better readability
Add Text Labels:
The for loop iterates over each bar, adding a text label on top of each bar showing the exact movie runtime.
plt.text() function is used to place the text.
Adjust Layout * plt.tight_layout(): Adjusts the layout so that the labels and titles do not overlap.
Display Plot:
plt.show(): Displays the generated plot.
To use this code:

Make sure you have matplotlib installed (pip install matplotlib).
Save the code as a .py file (e.g., movie_runtimes.py).
Run the file from your terminal: python movie_runtimes.py
This will display a bar chart showing the runtimes of the last 10 Denis Villeneuve movies.

Denis Villeneuve last 10 movies Denis Villeneuve movies runtimes How to plot movie runtimes with python matplotlib
First review the supplied code to make sure it does what you expect, then copy it here to try out the chart.


import matplotlib.pyplot as plt

# Movie titles and runtimes (in minutes).
# Data gathered from search results.
movies = {
    "Dune: Part Two": 166,
    "Dune": 155,
    "Blade Runner 2049": 164,
    "Arrival": 116,
    "Sicario": 121,
    "Enemy": 91,
    "Prisoners": 153,
    "Incendies": 131,
    "Polytechnique": 77,
    "Maelström": 87
}

movie_titles = list(movies.keys())
runtimes = list(movies.values())

# Create the bar chart
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
plt.bar(movie_titles, runtimes, color='skyblue')

# Add labels and title
plt.xlabel("Movie Title", fontsize=12)
plt.ylabel("Runtime (minutes)", fontsize=12)
plt.title("Denis Villeneuve Movie Runtimes", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10) # Rotate x-axis labels for better fit
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# Show the plot
plt.show()
     

One feature of using a chat conversation to do this is that you can now ask the model to make changes.


r = movie_chat.send_message('Looks great! Can you give the chart a dark theme instead?')
show_parts(r)
     
Okay, here's the modified code to give the chart a dark theme:

import matplotlib.pyplot as plt

# Movie titles and runtimes in minutes
movies = [
    "Dune: Part Two", "Dune: Part One", "Blade Runner 2049", "Arrival", "Sicario",
    "Enemy", "Prisoners", "Incendies", "Polytechnique", "Maelström"
]
runtimes = [166, 155, 164, 116, 121, 91, 153, 131, 77, 87]


# Set the dark theme
plt.style.use('dark_background')

# Create the bar plot
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
plt.bar(movies, runtimes, color='skyblue')

# Add labels and title with white color for better contrast
plt.xlabel("Movies", color='white', fontsize = 12)
plt.ylabel("Runtime (minutes)", color='white', fontsize = 12)
plt.title("Denis Villeneuve Movie Runtimes (Last 10 Movies)", color='white', fontsize = 14)
plt.xticks(rotation=45, ha="right", color='white')
plt.yticks(color='white')

# Add the exact runtime on top of each bar
for i, runtime in enumerate(runtimes):
    plt.text(i, runtime + 1, str(runtime), ha='center', va='bottom', color='white')


# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Display the plot
plt.show()
Changes:

plt.style.use('dark_background'): This line sets the overall plot style to use a dark background, providing a dark theme.
Label and Title Colors:
Added color='white' to plt.xlabel(), plt.ylabel(), plt.title(), plt.xticks() and plt.yticks() to ensure that the labels and titles are clearly visible on the dark background.
Text Label Color:
Added color='white' to plt.text() to make sure the text on top of the bars is also visible.
Now when you run this code, the chart will have a dark background with white text, creating a dark theme.

Again, always be sure to review code generated by the model before running it.


import matplotlib.pyplot as plt

# Movie titles and runtimes (in minutes).
# Data gathered from search results.
movies = {
    "Dune: Part Two": 166,
    "Dune": 155,
    "Blade Runner 2049": 164,
    "Arrival": 116,
    "Sicario": 121,
    "Enemy": 91,
    "Prisoners": 153,
    "Incendies": 131,
    "Polytechnique": 77,
    "Maelström": 87
}

movie_titles = list(movies.keys())
runtimes = list(movies.values())

# Set dark theme
plt.style.use('dark_background')

# Create the bar chart
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
plt.bar(movie_titles, runtimes, color='lightseagreen') #changed bar color

# Add labels and title
plt.xlabel("Movie Title", fontsize=12, color='white')
plt.ylabel("Runtime (minutes)", fontsize=12, color='white')
plt.title("Denis Villeneuve Movie Runtimes", fontsize=14, color='white')
plt.xticks(rotation=45, ha="right", fontsize=10, color='white') # Rotate x-axis labels for better fit
plt.yticks(color='white')
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

# Show the plot
plt.show()

     

Use search in the Multimodal Live API
The Search tool can be used in a live streaming context to have the model formulate grounded responses during the conversation.

Define some helpers
To use the bi-directional streaming API in Colab, you will buffer the audio stream. Define a play_response helper function to do the buffering, and once the audio for the current turn has completed, display an IPython audio widget.

As each of the following examples only use a single prompt, also define a run helper to wrap the setup and prompt execution steps into a single function call. This helper takes a config argument that will be added to the generation_config, so that you can toggle the Search tool between examples.


import asyncio
import io
import json
import re
import time
import wave

import numpy as np
from IPython.display import Audio, display


DEFAULT_OUTPUT_RATE = 24000
BASE_MODEL_CONFIG = {
    'generation_config' : {
        # Here you can change the model's output mode between either audio or text.
        # While this code expects an audio stream, text should work, but the stream
        # may interleave with the `Buffering....` text.
        'response_modalities': ['AUDIO']
    },
}

async def play_response(stream):
  """Buffer audio output and display a widget. Returns the streamed responses."""
  turn_buf = io.BytesIO()
  sample_rate = DEFAULT_OUTPUT_RATE

  all_responses = []

  print('Buffering', end='')
  async for msg in stream.receive():
    all_responses.append(msg)

    if text:=msg.text:
      print(text)
    if audio_data := msg.data:
      turn_buf.write(audio_data)
      if m := re.search(
          'rate=(?P\d+)',
          msg.server_content.model_turn.parts[0].inline_data.mime_type
      ):
            sample_rate = int(m.group('rate'))

    elif tool_call := msg.tool_call:
      # Handle tool-call requests. Here is where you would implement
      # custom tool code, but for this example, all tools respond 'ok'.
      for fc in tool_call.function_calls:
        print('Tool call', end='')
        tool_response = genai.types.LiveClientToolResponse(
            function_responses=[genai.types.FunctionResponse(
                name=fc.name,
                id=fc.id,
                response={'result': 'ok'},
            )]
        )
        await stream.send(tool_response)

    print('.', end='')

  print()

  # Play the audio
  if turn_buf.tell():
    audio = np.frombuffer(turn_buf.getvalue(), dtype=np.int16)
    display(Audio(audio, autoplay=True, rate=sample_rate))
  else:
    print('No audio :(')
    print(f'  {len(all_responses)=}')

  return all_responses


async def run(query, config=None):
  # Add any tools or other generation config.
  config = BASE_MODEL_CONFIG | (config or {})

  # Establish a live session. While this context manager is active, the
  # conversation will continue.
  async with client.aio.live.connect(model=MODEL, config=config) as strm:

    # Send the prompt.
    await strm.send(query, end_of_turn=True)
    # Handle the model response.
    responses = await play_response(strm)

    return responses
     
Stream with the Search tool
First, execute a query without the Search tool to observe the model's response to a time-sensitive query.

Note that the Multimodal Live API is a 2-way streaming API, but to simplify running in a notebook, each audio response is buffered and played once it has been fully streamed, so you will need to wait a few seconds before the response starts to play.


await run('Who won the skateboarding gold medals in the 2024 olympics?');
     
Buffering............................................
Now re-run with the Search tool enabled.


responses = await run('Who won the skateboarding gold medals in the 2024 olympics?', {'tools': [search_tool]})
     
Buffering.......................................................................................................................................................................
If you wish to see the full output that was returned, you can enable show_output here and run this cell. It includes the complete audio binary data, so it is off by default.


show_output = False

if show_output:
  for msg in responses:
    print(msg.model_dump(exclude_none=True))
     
Search with custom tools
In the Multimodal Live API, the Search tool can be used in conjunction with other tools, including function calls that you provide to the model.

In this example, you define a function set_climate that takes 2 parameters, mode (hot, cold, etc) and strength (0-10), and ask the model to set the climate control based on the live weather in the location you specify.


set_climate_tool = {'function_declarations': [{
    'name': 'set_climate',
    'description': 'Switches the local climate control equipment to the specified parameters.',
    'parameters': {
      'type': 'OBJECT',
      'properties': {
        # Define the "mode" argument.
        'mode': {
            'type': 'STRING',
            'enum': [
              # Define the possible values for "mode".
              "hot",
              "cold",
              "fan",
              "off",
            ],
            'description': 'Mode for the climate unit - whether to heat, cool or just blow air.',
        },
        # Define the "strength" argument.
        'strength': {
            'type': 'INTEGER',
            'description': 'Intensity of the climate to apply, 0-10 (0 is off, 10 is MAX).',
        },
      },
    },
  },
]}

search_tool = {'google_search': {}}

tools = {'tools': [search_tool, set_climate_tool]}

responses = await run("Look up the weather in Paris and set my climate control appropriately.", tools)
     
Buffering............Tool call.......................................................................................
Now inspect the tool_call response(s) you received during the conversation.


for r in responses:
  if tool := r.tool_call:
    for fn in tool.function_calls:
      args = ', '.join(f'{k}={v}' for k, v in fn.args.items())
      print(f'{fn.name}({args})  # id={fn.id}')
     
set_climate(strength=7, mode=hot)  # id=function-call-6271898767513010323


Another search notebook:

{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lb5yiH5h8x3h"
      },
      "source": [
        "##### Copyright 2024 Google LLC."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "cellView": "form",
        "id": "906e07f6e562"
      },
      "outputs": [],
      "source": [
        "#@title Licensed under the Apache License, Version 2.0 (the \"License\");\n",
        "# you may not use this file except in compliance with the License.\n",
        "# You may obtain a copy of the License at\n",
        "#\n",
        "# https://www.apache.org/licenses/LICENSE-2.0\n",
        "#\n",
        "# Unless required by applicable law or agreed to in writing, software\n",
        "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
        "# See the License for the specific language governing permissions and\n",
        "# limitations under the License."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1YXR7Yn480fU"
      },
      "source": [
        "# Search tool with Gemini 2.0"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CeJyB7rG82ph"
      },
      "source": [
        "<table align=\"left\">\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Search_grounding_for_research_report.ipynb\"><img src=\"../images/colab_logo_32px.png\" />Run in Google Colab</a>\n",
        "  </td>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "___eV40o8399"
      },
      "source": [
        "In this tutorial you are going to leverage the latest [search tool](https://ai.google.dev/gemini-api/docs/models/gemini-v2#search-tool) of the Gemini 2.0 model to write a company report. Note that the search tool is a paid ony feature and this tutorial does not work with a free tier API key.\n",
        "\n",
        "You may be asking, why does one need to use the search tool for this purpose? Well, as you may be aware, today's business world evolves very fast and LLMs generally are not trained frequently enough to capture the latest updates. Luckily Google search comes to the rescue. Google search is built to provide accurate and nearly realtime information and can help us fulfill this task perfectly.\n",
        "\n",
        "Note that while Gemini 1.5 models offered the [search grounding](https://ai.google.dev/gemini-api/docs/grounding) feature which may also help you achieve similar results (see [a previous notebook](https://github.com/google-gemini/cookbook/blob/gemini-1.5-archive/examples/Search_grounding_for_research_report.ipynb) using search grounding), the new search tool in Gemini 2.0 models is much more powerful and easier to use, and should be prioritized over search grounding. It is also possible to use multiple tools together (for example, search tool and function calling)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Mfk6YY3G5kqp"
      },
      "source": [
        "## Setup"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "d5027929de8f"
      },
      "source": [
        "### Install SDK\n",
        "\n",
        "The new **[Google Gen AI SDK](https://ai.google.dev/gemini-api/docs/sdks)** provides programmatic access to Gemini 2.0 (and previous models) using both the [Google AI for Developers](https://ai.google.dev/gemini-api/docs) and [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview) APIs. With a few exceptions, code that runs on one platform will run on both. This means that you can prototype an application using the Developer API and then migrate the application to Vertex AI without rewriting your code.\n",
        "\n",
        "More details about this new SDK on the [documentation](https://ai.google.dev/gemini-api/docs/sdks) or in the [Getting started](../gemini-2/get_started.ipynb) notebook."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "46zEFO2a9FFd"
      },
      "outputs": [],
      "source": [
        "!pip install -U -q google-genai"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CTIfnvCn9HvH"
      },
      "source": [
        "### Setup your API key\n",
        "\n",
        "To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "A1pkoyZb9Jm3"
      },
      "outputs": [],
      "source": [
        "from google.colab import userdata\n",
        "\n",
        "GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QclzsisW0nTK"
      },
      "source": [
        "### Import the libraries"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cXNAAIqOuXlO"
      },
      "outputs": [],
      "source": [
        "from google import genai\n",
        "from google.genai.types import GenerateContentConfig, Tool\n",
        "from IPython.display import display, HTML, Markdown\n",
        "import io\n",
        "import json\n",
        "import re"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3Hx_Gw9i0Yuv"
      },
      "source": [
        "### Initialize SDK client\n",
        "\n",
        "With the new SDK you now only need to initialize a client with you API key (or OAuth if using [Vertex AI](https://cloud.google.com/vertex-ai)). The model is now set in each call."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HghvVpbU0Uap"
      },
      "outputs": [],
      "source": [
        "client = genai.Client(api_key=GOOGLE_API_KEY)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QOov6dpG99rY"
      },
      "source": [
        "### Select a model\n",
        "\n",
        "Search tool is a new feature in the Gemini 2.0 model that automatically retrieves accurate and grounded artifacts from the web for developers to further process. Unlike the search grounding in the Gemini 1.5 models, you do not need to set the dynamic retrieval threshold.\n",
        "\n",
        "For more information about all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for extended information on each of them.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "27Fikag0xSaB"
      },
      "outputs": [],
      "source": [
        "MODEL_ID = \"gemini-2.0-flash\" # @param [\"gemini-1.5-flash-latest\",\"gemini-2.0-flash-lite-preview-02-05\",\"gemini-2.0-flash\",\"gemini-2.0-pro-exp-02-05\"] {\"allow-input\":true}"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "F9lFY3h6unY2"
      },
      "source": [
        "## Write the report with Gemini 2.0"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "j562CvvWvASd"
      },
      "source": [
        "### Select a target company to research\n",
        "\n",
        "Next you will use Alphabet as an example research target."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7UQxJr7UvEhl"
      },
      "outputs": [],
      "source": [
        "COMPANY = 'Alphabet' # @param {type:\"string\"}"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HALOE8He78H1"
      },
      "source": [
        "### Have Gemini 2.0 plan for the task and write the report\n",
        "\n",
        "Gemini 2.0 is a huge step up from previous models since it can reason, plan, search and write in one go. In this case, you will only give Gemini 2.0 a prompt to do all of these and the model will finish your task seamlessly. You will also using output streaming, which streams the response as it is being generated, and the model will return chunks of the response as soon as they are generated. If you would like the SDK to return a response after the model completes the entire generation process,  you can use the non-streaming approach as well.\n",
        "\n",
        "Note that you must enable Google Search Suggestions, which help users find search results corresponding to a grounded response, and [display the search queries](https://ai.google.dev/gemini-api/docs/grounding/search-suggestions#display-requirements) that are included in the grounded response's metadata. You can find [more details](https://ai.google.dev/gemini-api/terms#grounding-with-google-search) about this requirement."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IAl2UGtw0LPG"
      },
      "outputs": [
        {
          "data": {
            "text/markdown": [
              "Okay"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              ", I will research Alphabet and create a company report. Here's my plan"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              ":\n",
              "\n",
              "1.  **Company Overview:** I'll start by gathering basic"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " information about Alphabet, such as its founding date, headquarters, and what it does.\n",
              "2.  **Business Segments:** I'll identify the different"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " business segments that Alphabet operates in (e.g., Google Services, Google Cloud, Other Bets).\n",
              "3.  **Recent News and Developments:** I'"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "ll look for recent news articles and press releases to understand any significant updates, product launches, or strategic shifts.\n",
              "4.  **Financial Performance:** I'll try to find information about Alphabet's recent financial performance, including revenue,"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " profit, and key metrics.\n",
              "5.  **Stock Information:** I'll check the current stock price and any recent stock-related news.\n",
              "6.  **Key People:** I'll identify key executives and board members."
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "\n",
              "7.  **Competitive Landscape:** I'll briefly touch on the competitive landscape and Alphabet's main competitors.\n",
              "\n",
              "Now, let's start with the searches.\n",
              "\n"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "Okay"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              ", I've gathered a lot of information about Alphabet. Here's a"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " company report:\n",
              "\n",
              "---\n",
              "**Company Report: Alphabet Inc.**\n",
              "\n",
              "**Overview"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              ":**\n",
              "\n",
              "Alphabet Inc. is a multinational technology conglomerate holding company headquartered in Mountain View, California. It was created through a restructuring of Google on October 2,"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " 2015, becoming the parent company of Google and several former Google subsidiaries. Alphabet is the world's second-largest technology company by revenue,"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " after Apple, and one of the world's most valuable companies. It is considered one of the Big Five American information technology companies, alongside Amazon, Apple, Meta, and Microsoft.\n",
              "\n",
              "**Founding and Restructuring:**\n",
              "\n",
              "While Google"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " was founded in 1998 by Larry Page and Sergey Brin, Alphabet Inc. was formed on October 2, 2015, as a restructuring to make the core Google business \"cleaner and more accountable\""
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " while allowing greater autonomy to other ventures. In December 2019, Larry Page and Sergey Brin stepped down from their executive roles, and Sundar Pichai, who was already CEO of Google, became CEO of both Google and Alphabet. Page and Brin remain employees, board members, and controlling shareholders."
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "\n",
              "\n",
              "**Business Segments:**\n",
              "\n",
              "Alphabet operates through three main reportable segments:\n",
              "\n",
              "*   **Google Services:** This segment includes Google's core internet products such as ads, Android, Chrome, hardware, Google Cloud, Google Maps, Google Play, Search, and YouTube. This is the most profitable segment, with"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " revenue primarily generated from advertising on Google Search and YouTube, as well as subscriptions and smartphone sales.\n",
              "*   **Google Cloud:** This segment provides cloud services for businesses, including office software and computing platforms. Google Cloud's quarterly revenues have recently surpassed $10 billion, and it is showing strong growth.\n",
              "*"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "   **Other Bets:** This segment includes various cutting-edge technology innovation ventures at different stages of development, such as Access, Calico, CapitalG, GV, Verily, Waymo, and X. This segment often posts operating losses.\n",
              "\n",
              "**Recent News and Developments:**\n",
              "\n",
              "*   Alphabet's stock recently"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " reached an all-time high after unveiling its latest AI model, Gemini 2.0, and showcasing a new quantum computing chip called Willow.\n",
              "*   Google has launched Gemini, formerly known as Bard, an AI chatbot that supports over 40 languages.\n",
              "*   Alphabet's Q3 20"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "24 results showed a 15% increase in consolidated revenue year-over-year, reaching $88.3 billion. Google Services revenue increased by 13% to $76.5 billion.\n",
              "*   Google Cloud's revenue exceeded $10 billion in a recent quarter, with over"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " $1 billion in operating profit.\n",
              "*   Alphabet announced a cash dividend of $0.20 per share in July 2024.\n",
              "\n",
              "**Financial Performance:**\n",
              "\n",
              "*   In Q3 2024, Alphabet's consolidated revenue was $88.3 billion, a 1"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "5% increase year-over-year.\n",
              "*   Net income for Q3 2024 was $26.3 billion, compared to $19.7 billion a year ago.\n",
              "*   For the nine months ended September 30, 2024, sales were $"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "253.5 billion, compared to $221.1 billion a year ago.\n",
              "*   Google Cloud's revenue has surpassed $10 billion in a recent quarter.\n",
              "*   Alphabet's revenue is primarily driven by Google Services, particularly advertising.\n",
              "\n",
              "**Stock Information:**\n",
              "\n",
              "*   "
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "Alphabet has two classes of stock: GOOG (Class C shares without voting rights) and GOOGL (Class A common stock).\n",
              "*   As of today, December 17, 2024, the price of GOOG is around $198.16, and it has increased"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " by 4.31% in the past 24 hours.\n",
              "*   The stock reached an all-time high on December 10, 2024.\n",
              "*   Analysts' price targets for GOOG range from $185 to $240.\n",
              "\n",
              "**Key People"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              ":**\n",
              "\n",
              "*   **CEO:** Sundar Pichai\n",
              "*   **Co-Founders:** Larry Page and Sergey Brin (remain board members and controlling shareholders)\n",
              "*   **Chief Financial Officer:** Anat Ashkenazi\n",
              "*   **Other Key Executives:** Philipp Schindler (Chief Business Officer, Google), Prabhakar"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              " Raghavan (Chief Technologist, Google), Ruth M. Porat (President and Chief Investment Officer; CFO)\n",
              "\n",
              "**Competitive Landscape:**\n",
              "\n",
              "Alphabet's main competitors include:\n",
              "\n",
              "*   **Other Big Tech Companies:** Apple, Microsoft, Meta, Amazon, IBM\n",
              "*   **Software Companies:** SAP, Pal"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "antir Technologies, Shopify, AppLovin, Infosys, CrowdStrike, Atlassian, Trade Desk, NetEase, Snowflake\n",
              "*   **Other Competitors:** ByteDance, Zoom, Salesforce, Tencent, Adobe\n",
              "\n",
              "**Additional Information:**\n",
              "\n",
              "*   Alphabet's headquarters is located at 160"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              "0 Amphitheatre Parkway in Mountain View, California, also known as the Googleplex.\n",
              "*   Alphabet has a global presence with employees across six continents.\n",
              "*   Alphabet is involved in investing in infrastructure, data management, analytics, and artificial intelligence (AI).\n",
              "\n",
              "This report provides a comprehensive overview of Alphabet Inc"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/markdown": [
              ". as of December 17, 2024.\n",
              "---\n"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "<style>\n",
              ".container {\n",
              "  align-items: center;\n",
              "  border-radius: 8px;\n",
              "  display: flex;\n",
              "  font-family: Google Sans, Roboto, sans-serif;\n",
              "  font-size: 14px;\n",
              "  line-height: 20px;\n",
              "  padding: 8px 12px;\n",
              "}\n",
              ".chip {\n",
              "  display: inline-block;\n",
              "  border: solid 1px;\n",
              "  border-radius: 16px;\n",
              "  min-width: 14px;\n",
              "  padding: 5px 16px;\n",
              "  text-align: center;\n",
              "  user-select: none;\n",
              "  margin: 0 8px;\n",
              "  -webkit-tap-highlight-color: transparent;\n",
              "}\n",
              ".carousel {\n",
              "  overflow: auto;\n",
              "  scrollbar-width: none;\n",
              "  white-space: nowrap;\n",
              "  margin-right: -12px;\n",
              "}\n",
              ".headline {\n",
              "  display: flex;\n",
              "  margin-right: 4px;\n",
              "}\n",
              ".gradient-container {\n",
              "  position: relative;\n",
              "}\n",
              ".gradient {\n",
              "  position: absolute;\n",
              "  transform: translate(3px, -9px);\n",
              "  height: 36px;\n",
              "  width: 9px;\n",
              "}\n",
              "@media (prefers-color-scheme: light) {\n",
              "  .container {\n",
              "    background-color: #fafafa;\n",
              "    box-shadow: 0 0 0 1px #0000000f;\n",
              "  }\n",
              "  .headline-label {\n",
              "    color: #1f1f1f;\n",
              "  }\n",
              "  .chip {\n",
              "    background-color: #ffffff;\n",
              "    border-color: #d2d2d2;\n",
              "    color: #5e5e5e;\n",
              "    text-decoration: none;\n",
              "  }\n",
              "  .chip:hover {\n",
              "    background-color: #f2f2f2;\n",
              "  }\n",
              "  .chip:focus {\n",
              "    background-color: #f2f2f2;\n",
              "  }\n",
              "  .chip:active {\n",
              "    background-color: #d8d8d8;\n",
              "    border-color: #b6b6b6;\n",
              "  }\n",
              "  .logo-dark {\n",
              "    display: none;\n",
              "  }\n",
              "  .gradient {\n",
              "    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);\n",
              "  }\n",
              "}\n",
              "@media (prefers-color-scheme: dark) {\n",
              "  .container {\n",
              "    background-color: #1f1f1f;\n",
              "    box-shadow: 0 0 0 1px #ffffff26;\n",
              "  }\n",
              "  .headline-label {\n",
              "    color: #fff;\n",
              "  }\n",
              "  .chip {\n",
              "    background-color: #2c2c2c;\n",
              "    border-color: #3c4043;\n",
              "    color: #fff;\n",
              "    text-decoration: none;\n",
              "  }\n",
              "  .chip:hover {\n",
              "    background-color: #353536;\n",
              "  }\n",
              "  .chip:focus {\n",
              "    background-color: #353536;\n",
              "  }\n",
              "  .chip:active {\n",
              "    background-color: #464849;\n",
              "    border-color: #53575b;\n",
              "  }\n",
              "  .logo-light {\n",
              "    display: none;\n",
              "  }\n",
              "  .gradient {\n",
              "    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);\n",
              "  }\n",
              "}\n",
              "</style>\n",
              "<div class=\"container\">\n",
              "  <div class=\"headline\">\n",
              "    <svg class=\"logo-light\" width=\"18\" height=\"18\" viewBox=\"9 9 35 35\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">\n",
              "      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z\" fill=\"#4285F4\"/>\n",
              "      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z\" fill=\"#34A853\"/>\n",
              "      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z\" fill=\"#FBBC05\"/>\n",
              "      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z\" fill=\"#EA4335\"/>\n",
              "    </svg>\n",
              "    <svg class=\"logo-dark\" width=\"18\" height=\"18\" viewBox=\"0 0 48 48\" xmlns=\"http://www.w3.org/2000/svg\">\n",
              "      <circle cx=\"24\" cy=\"23\" fill=\"#FFF\" r=\"22\"/>\n",
              "      <path d=\"M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z\" fill=\"#4285F4\"/>\n",
              "      <path d=\"M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z\" fill=\"#34A853\"/>\n",
              "      <path d=\"M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z\" fill=\"#FBBC05\"/>\n",
              "      <path d=\"M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z\" fill=\"#EA4335\"/>\n",
              "    </svg>\n",
              "    <div class=\"gradient-container\"><div class=\"gradient\"></div></div>\n",
              "  </div>\n",
              "  <div class=\"carousel\">\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcR-71ru6mpc4gpy8vuvnMTkxTbdFKa8j3faKpVo5-JDZsZ5vP77tulubZLb9XPFYIwoPVi0lxXNQI1VIB4728Vor6SDpVTgFzvcoyI_XV4Fclmf0JoIzi4SxnwJJR-t-khsRV1zIoF6JFjAt7vDb_Wk5l9m2-VfPxeTglYvr3VPnLKcQArFkF6KUhM0N4OGZsVsHgYSo3AijYtqmoA=\">Alphabet financial performance</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcReU_kkKN1xoCrHouZzAdvCiMEqGQyE3w8hBExnm9AZtyj8Cjb--Ajhm9qKDVFCPzOlcHviSjHD26Oxjwx6cCF19Xz2lYCi86GP4e8rr3qxi17Z_ySzxtlHjU_57FabN-6OI94SBWPoF9e8WxhfJZejUlQhhI8xDywEXYXaCWAi8_HCDPnWznEEgiEoQpDaOTSsgAj0V06N835ZRQ==\">Alphabet company headquarters</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcS38VqBImrlUHA3ZpnsDyrwmH5lnFCsL80ahp6MycxavLqx9LpU_LOUGlyo44Iltmcu7soEBs9QkwYfYUIv7bqM4w6NHbWp5K-5JuYjfW8qO9xG243pMoN591O6cio6tlMqpgIMjvPzYC0diIvSfyhwgDmzoyJ9Wpa2-PUEdk93O4vsA4IS56FJC4QgBTaulCXUj9PCruQ=\">what is Alphabet company</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcQcggyxRwNLHNMDjO_UavX_aND9VWXm1H7f9O0yQTUmX85P8R8rxbIAyotz3F34oUbdlsHoCgy9GjepRCovpw2Z3dlw-7W1VOzpgsb1FDNnMp1fLcB0k2QVU94c8Vi7z_iPbNjUWRc6uxNU3PQMw9sF13sPlaExHVe6sUGnLs-9uIAFt9IoBUD5lcLMt18uSoqD_jSQ9AN06ysJb40=\">Alphabet company founding date</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcToBnSGGzKTSYx6VfCsa6K6jxUzvztDKktQaShAzPfMomF-e5kvNJurZSXf6ET4bE04xROQCR_z9ldbkVZXw2nu2623Tk6c1xnjouHzOqrbk1-p1JOcya1sLprjiEJEzMlw3BnpYW4n34Ol_kT7oiApKKKiOn6SP6OANDrVasUqAkTp6VBTX2rI1zxVaobdt8cyZQ==\">Alphabet recent news</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcSz9jwXP459pNRQCwaOf7INS_GmByAis8vCQ6VVCGjlBKTwdibbCp9h1L9S8x8I_SdvhcmfTiYclqsSIbgN5I6d2dNT4ITQIBdLhOvUqmCNv23t3MYNCw7Po6j_43wOJ9c-hHMoCrQUuwfxxyi5p4SXaaQblqBZVn_vkoTIMSUe7XovJLfFyV72sETwe-aIUJVrfw==\">Alphabet stock price</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcRDdy7gKzgF3aeuTV63Xq6zpbK78V5Got-eHzPYYtLr_ku6BnRsYns_s8-1xHogTgLkFGSR9saLKPgA8pDvaGlhbguWfHC-to2yyxrSKPQogKazcGAW9rowzlh-bV-E-GFXGTy4quIGVN4holwR7pfu4WtCJZGBBDl_Z5gUs1RQVqlpfqYW4fNOnv9GsFF25UknCOP81NKQsg==\">Alphabet business segments</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcQy3R-P35aHSfcS76PmdshLmomshcU76g6XlciWqoHAqI1WFCbL2JnPQrET7QXrTAey9dPpXNnnj3hEt-nsnoOUpJoKSQuvXm0vkOXqHax_XF-4i2ZUdWNDcu6I3XMShdla6qQQ8DGS3u1bcbfR34g2RHxTq8sczJVHwRLTGGqA1M4vkWcrI8-A_aMhdmgzMSAiTw==\">Alphabet competitors</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcQ7ih2X07lFdtLZJ24dIcc4EQimM7_r34KVbbCqGq74ahvn1kwasBP3q3avmXRpTPv-yr3NFHGF7ZNB2WRocNhYiigiRlGLmJ0yVwj8I4HUQFP_vUsmpGwl3Y4wF89q2C7KIO_WPnld8UXF4GGVjyp-cEi1T0H6if80QP1T8MqBtEviROpApvvXy1dxa3Ec4T5ftUtShQ==\">Alphabet key executives</a>\n",
              "  </div>\n",
              "</div>\n"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "sys_instruction = \"\"\"You are an analyst that conducts company research.\n",
        "You are given a company name, and you will work on a company report. You have access\n",
        "to Google Search to look up company news, updates and metrics to write research reports.\n",
        "\n",
        "When given a company name, identify key aspects to research, look up that information\n",
        "and then write a concise company report.\n",
        "\n",
        "Feel free to plan your work and talk about it, but when you start writing the report,\n",
        "put a line of dashes (---) to demarcate the report itself, and say nothing else after\n",
        "the report has finished.\n",
        "\"\"\"\n",
        "\n",
        "config = GenerateContentConfig(system_instruction=sys_instruction, tools=[Tool(google_search={})], temperature=0)\n",
        "response_stream = client.models.generate_content_stream(\n",
        "    model=MODEL, config=config, contents=[COMPANY])\n",
        "\n",
        "report = io.StringIO()\n",
        "for chunk in response_stream:\n",
        "  candidate = chunk.candidates[0]\n",
        "\n",
        "  for part in candidate.content.parts:\n",
        "    if part.text:\n",
        "      display(Markdown(part.text))\n",
        "\n",
        "      # Find and save the report itself.\n",
        "      if m := re.search('(^|\\n)-+\\n(.*)$', part.text, re.M):\n",
        "          # Find the starting '---' line and start saving.\n",
        "          report.write(m.group(2))\n",
        "      elif report.tell():\n",
        "        # If there's already something in the buffer, keep recording.\n",
        "        report.write(part.text)\n",
        "\n",
        "    else:\n",
        "      print(json.dumps(part.model_dump(exclude_none=True), indent=2))\n",
        "\n",
        "  # You must enable Google Search Suggestions\n",
        "  if gm := candidate.grounding_metadata:\n",
        "    if sep := gm.search_entry_point:\n",
        "      display(HTML(sep.rendered_content))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "e2gBbhEU-jCo"
      },
      "source": [
        "Very impressive! Gemini 2.0 starts by planning for the task, performing relevant searches and streams out a report for you."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "P_JoMFa18Rss"
      },
      "source": [
        "### Final output\n",
        "\n",
        "Render the final output."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1yddCUCL4yKA"
      },
      "outputs": [
        {
          "data": {
            "text/markdown": [
              "**Company Report: Alphabet Inc.**:**\n",
              "\n",
              "Alphabet Inc. is a multinational technology conglomerate holding company headquartered in Mountain View, California. It was created through a restructuring of Google on October 2, 2015, becoming the parent company of Google and several former Google subsidiaries. Alphabet is the world's second-largest technology company by revenue, after Apple, and one of the world's most valuable companies. It is considered one of the Big Five American information technology companies, alongside Amazon, Apple, Meta, and Microsoft.\n",
              "\n",
              "**Founding and Restructuring:**\n",
              "\n",
              "While Google was founded in 1998 by Larry Page and Sergey Brin, Alphabet Inc. was formed on October 2, 2015, as a restructuring to make the core Google business \"cleaner and more accountable\" while allowing greater autonomy to other ventures. In December 2019, Larry Page and Sergey Brin stepped down from their executive roles, and Sundar Pichai, who was already CEO of Google, became CEO of both Google and Alphabet. Page and Brin remain employees, board members, and controlling shareholders.\n",
              "\n",
              "**Business Segments:**\n",
              "\n",
              "Alphabet operates through three main reportable segments:\n",
              "\n",
              "*   **Google Services:** This segment includes Google's core internet products such as ads, Android, Chrome, hardware, Google Cloud, Google Maps, Google Play, Search, and YouTube. This is the most profitable segment, with revenue primarily generated from advertising on Google Search and YouTube, as well as subscriptions and smartphone sales.\n",
              "*   **Google Cloud:** This segment provides cloud services for businesses, including office software and computing platforms. Google Cloud's quarterly revenues have recently surpassed \\$10 billion, and it is showing strong growth.\n",
              "*   **Other Bets:** This segment includes various cutting-edge technology innovation ventures at different stages of development, such as Access, Calico, CapitalG, GV, Verily, Waymo, and X. This segment often posts operating losses.\n",
              "\n",
              "**Recent News and Developments:**\n",
              "\n",
              "*   Alphabet's stock recently reached an all-time high after unveiling its latest AI model, Gemini 2.0, and showcasing a new quantum computing chip called Willow.\n",
              "*   Google has launched Gemini, formerly known as Bard, an AI chatbot that supports over 40 languages.\n",
              "*   Alphabet's Q3 2024 results showed a 15% increase in consolidated revenue year-over-year, reaching \\$88.3 billion. Google Services revenue increased by 13% to \\$76.5 billion.\n",
              "*   Google Cloud's revenue exceeded \\$10 billion in a recent quarter, with over \\$1 billion in operating profit.\n",
              "*   Alphabet announced a cash dividend of \\$0.20 per share in July 2024.\n",
              "\n",
              "**Financial Performance:**\n",
              "\n",
              "*   In Q3 2024, Alphabet's consolidated revenue was \\$88.3 billion, a 15% increase year-over-year.\n",
              "*   Net income for Q3 2024 was \\$26.3 billion, compared to \\$19.7 billion a year ago.\n",
              "*   For the nine months ended September 30, 2024, sales were \\$253.5 billion, compared to \\$221.1 billion a year ago.\n",
              "*   Google Cloud's revenue has surpassed \\$10 billion in a recent quarter.\n",
              "*   Alphabet's revenue is primarily driven by Google Services, particularly advertising.\n",
              "\n",
              "**Stock Information:**\n",
              "\n",
              "*   Alphabet has two classes of stock: GOOG (Class C shares without voting rights) and GOOGL (Class A common stock).\n",
              "*   As of today, December 17, 2024, the price of GOOG is around \\$198.16, and it has increased by 4.31% in the past 24 hours.\n",
              "*   The stock reached an all-time high on December 10, 2024.\n",
              "*   Analysts' price targets for GOOG range from \\$185 to \\$240.\n",
              "\n",
              "**Key People:**\n",
              "\n",
              "*   **CEO:** Sundar Pichai\n",
              "*   **Co-Founders:** Larry Page and Sergey Brin (remain board members and controlling shareholders)\n",
              "*   **Chief Financial Officer:** Anat Ashkenazi\n",
              "*   **Other Key Executives:** Philipp Schindler (Chief Business Officer, Google), Prabhakar Raghavan (Chief Technologist, Google), Ruth M. Porat (President and Chief Investment Officer; CFO)\n",
              "\n",
              "**Competitive Landscape:**\n",
              "\n",
              "Alphabet's main competitors include:\n",
              "\n",
              "*   **Other Big Tech Companies:** Apple, Microsoft, Meta, Amazon, IBM\n",
              "*   **Software Companies:** SAP, Palantir Technologies, Shopify, AppLovin, Infosys, CrowdStrike, Atlassian, Trade Desk, NetEase, Snowflake\n",
              "*   **Other Competitors:** ByteDance, Zoom, Salesforce, Tencent, Adobe\n",
              "\n",
              "**Additional Information:**\n",
              "\n",
              "*   Alphabet's headquarters is located at 1600 Amphitheatre Parkway in Mountain View, California, also known as the Googleplex.\n",
              "*   Alphabet has a global presence with employees across six continents.\n",
              "*   Alphabet is involved in investing in infrastructure, data management, analytics, and artificial intelligence (AI).\n",
              "\n",
              "This report provides a comprehensive overview of Alphabet Inc"
            ],
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "<style>\n",
              ".container {\n",
              "  align-items: center;\n",
              "  border-radius: 8px;\n",
              "  display: flex;\n",
              "  font-family: Google Sans, Roboto, sans-serif;\n",
              "  font-size: 14px;\n",
              "  line-height: 20px;\n",
              "  padding: 8px 12px;\n",
              "}\n",
              ".chip {\n",
              "  display: inline-block;\n",
              "  border: solid 1px;\n",
              "  border-radius: 16px;\n",
              "  min-width: 14px;\n",
              "  padding: 5px 16px;\n",
              "  text-align: center;\n",
              "  user-select: none;\n",
              "  margin: 0 8px;\n",
              "  -webkit-tap-highlight-color: transparent;\n",
              "}\n",
              ".carousel {\n",
              "  overflow: auto;\n",
              "  scrollbar-width: none;\n",
              "  white-space: nowrap;\n",
              "  margin-right: -12px;\n",
              "}\n",
              ".headline {\n",
              "  display: flex;\n",
              "  margin-right: 4px;\n",
              "}\n",
              ".gradient-container {\n",
              "  position: relative;\n",
              "}\n",
              ".gradient {\n",
              "  position: absolute;\n",
              "  transform: translate(3px, -9px);\n",
              "  height: 36px;\n",
              "  width: 9px;\n",
              "}\n",
              "@media (prefers-color-scheme: light) {\n",
              "  .container {\n",
              "    background-color: #fafafa;\n",
              "    box-shadow: 0 0 0 1px #0000000f;\n",
              "  }\n",
              "  .headline-label {\n",
              "    color: #1f1f1f;\n",
              "  }\n",
              "  .chip {\n",
              "    background-color: #ffffff;\n",
              "    border-color: #d2d2d2;\n",
              "    color: #5e5e5e;\n",
              "    text-decoration: none;\n",
              "  }\n",
              "  .chip:hover {\n",
              "    background-color: #f2f2f2;\n",
              "  }\n",
              "  .chip:focus {\n",
              "    background-color: #f2f2f2;\n",
              "  }\n",
              "  .chip:active {\n",
              "    background-color: #d8d8d8;\n",
              "    border-color: #b6b6b6;\n",
              "  }\n",
              "  .logo-dark {\n",
              "    display: none;\n",
              "  }\n",
              "  .gradient {\n",
              "    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);\n",
              "  }\n",
              "}\n",
              "@media (prefers-color-scheme: dark) {\n",
              "  .container {\n",
              "    background-color: #1f1f1f;\n",
              "    box-shadow: 0 0 0 1px #ffffff26;\n",
              "  }\n",
              "  .headline-label {\n",
              "    color: #fff;\n",
              "  }\n",
              "  .chip {\n",
              "    background-color: #2c2c2c;\n",
              "    border-color: #3c4043;\n",
              "    color: #fff;\n",
              "    text-decoration: none;\n",
              "  }\n",
              "  .chip:hover {\n",
              "    background-color: #353536;\n",
              "  }\n",
              "  .chip:focus {\n",
              "    background-color: #353536;\n",
              "  }\n",
              "  .chip:active {\n",
              "    background-color: #464849;\n",
              "    border-color: #53575b;\n",
              "  }\n",
              "  .logo-light {\n",
              "    display: none;\n",
              "  }\n",
              "  .gradient {\n",
              "    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);\n",
              "  }\n",
              "}\n",
              "</style>\n",
              "<div class=\"container\">\n",
              "  <div class=\"headline\">\n",
              "    <svg class=\"logo-light\" width=\"18\" height=\"18\" viewBox=\"9 9 35 35\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">\n",
              "      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z\" fill=\"#4285F4\"/>\n",
              "      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z\" fill=\"#34A853\"/>\n",
              "      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z\" fill=\"#FBBC05\"/>\n",
              "      <path fill-rule=\"evenodd\" clip-rule=\"evenodd\" d=\"M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z\" fill=\"#EA4335\"/>\n",
              "    </svg>\n",
              "    <svg class=\"logo-dark\" width=\"18\" height=\"18\" viewBox=\"0 0 48 48\" xmlns=\"http://www.w3.org/2000/svg\">\n",
              "      <circle cx=\"24\" cy=\"23\" fill=\"#FFF\" r=\"22\"/>\n",
              "      <path d=\"M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z\" fill=\"#4285F4\"/>\n",
              "      <path d=\"M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z\" fill=\"#34A853\"/>\n",
              "      <path d=\"M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z\" fill=\"#FBBC05\"/>\n",
              "      <path d=\"M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z\" fill=\"#EA4335\"/>\n",
              "    </svg>\n",
              "    <div class=\"gradient-container\"><div class=\"gradient\"></div></div>\n",
              "  </div>\n",
              "  <div class=\"carousel\">\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcR-71ru6mpc4gpy8vuvnMTkxTbdFKa8j3faKpVo5-JDZsZ5vP77tulubZLb9XPFYIwoPVi0lxXNQI1VIB4728Vor6SDpVTgFzvcoyI_XV4Fclmf0JoIzi4SxnwJJR-t-khsRV1zIoF6JFjAt7vDb_Wk5l9m2-VfPxeTglYvr3VPnLKcQArFkF6KUhM0N4OGZsVsHgYSo3AijYtqmoA=\">Alphabet financial performance</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcReU_kkKN1xoCrHouZzAdvCiMEqGQyE3w8hBExnm9AZtyj8Cjb--Ajhm9qKDVFCPzOlcHviSjHD26Oxjwx6cCF19Xz2lYCi86GP4e8rr3qxi17Z_ySzxtlHjU_57FabN-6OI94SBWPoF9e8WxhfJZejUlQhhI8xDywEXYXaCWAi8_HCDPnWznEEgiEoQpDaOTSsgAj0V06N835ZRQ==\">Alphabet company headquarters</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcS38VqBImrlUHA3ZpnsDyrwmH5lnFCsL80ahp6MycxavLqx9LpU_LOUGlyo44Iltmcu7soEBs9QkwYfYUIv7bqM4w6NHbWp5K-5JuYjfW8qO9xG243pMoN591O6cio6tlMqpgIMjvPzYC0diIvSfyhwgDmzoyJ9Wpa2-PUEdk93O4vsA4IS56FJC4QgBTaulCXUj9PCruQ=\">what is Alphabet company</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcQcggyxRwNLHNMDjO_UavX_aND9VWXm1H7f9O0yQTUmX85P8R8rxbIAyotz3F34oUbdlsHoCgy9GjepRCovpw2Z3dlw-7W1VOzpgsb1FDNnMp1fLcB0k2QVU94c8Vi7z_iPbNjUWRc6uxNU3PQMw9sF13sPlaExHVe6sUGnLs-9uIAFt9IoBUD5lcLMt18uSoqD_jSQ9AN06ysJb40=\">Alphabet company founding date</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcToBnSGGzKTSYx6VfCsa6K6jxUzvztDKktQaShAzPfMomF-e5kvNJurZSXf6ET4bE04xROQCR_z9ldbkVZXw2nu2623Tk6c1xnjouHzOqrbk1-p1JOcya1sLprjiEJEzMlw3BnpYW4n34Ol_kT7oiApKKKiOn6SP6OANDrVasUqAkTp6VBTX2rI1zxVaobdt8cyZQ==\">Alphabet recent news</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcSz9jwXP459pNRQCwaOf7INS_GmByAis8vCQ6VVCGjlBKTwdibbCp9h1L9S8x8I_SdvhcmfTiYclqsSIbgN5I6d2dNT4ITQIBdLhOvUqmCNv23t3MYNCw7Po6j_43wOJ9c-hHMoCrQUuwfxxyi5p4SXaaQblqBZVn_vkoTIMSUe7XovJLfFyV72sETwe-aIUJVrfw==\">Alphabet stock price</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcRDdy7gKzgF3aeuTV63Xq6zpbK78V5Got-eHzPYYtLr_ku6BnRsYns_s8-1xHogTgLkFGSR9saLKPgA8pDvaGlhbguWfHC-to2yyxrSKPQogKazcGAW9rowzlh-bV-E-GFXGTy4quIGVN4holwR7pfu4WtCJZGBBDl_Z5gUs1RQVqlpfqYW4fNOnv9GsFF25UknCOP81NKQsg==\">Alphabet business segments</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcQy3R-P35aHSfcS76PmdshLmomshcU76g6XlciWqoHAqI1WFCbL2JnPQrET7QXrTAey9dPpXNnnj3hEt-nsnoOUpJoKSQuvXm0vkOXqHax_XF-4i2ZUdWNDcu6I3XMShdla6qQQ8DGS3u1bcbfR34g2RHxTq8sczJVHwRLTGGqA1M4vkWcrI8-A_aMhdmgzMSAiTw==\">Alphabet competitors</a>\n",
              "    <a class=\"chip\" href=\"https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcQ7ih2X07lFdtLZJ24dIcc4EQimM7_r34KVbbCqGq74ahvn1kwasBP3q3avmXRpTPv-yr3NFHGF7ZNB2WRocNhYiigiRlGLmJ0yVwj8I4HUQFP_vUsmpGwl3Y4wF89q2C7KIO_WPnld8UXF4GGVjyp-cEi1T0H6if80QP1T8MqBtEviROpApvvXy1dxa3Ec4T5ftUtShQ==\">Alphabet key executives</a>\n",
              "  </div>\n",
              "</div>\n"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "display(Markdown(report.getvalue().replace('$', r'\\$')))  # Escape $ signs for better MD rendering\n",
        "display(HTML(sep.rendered_content))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "leOZmicAwTSn"
      },
      "source": [
        "As you can see, the Gemini 2.0 model is able to write a concise, accurate and well-structured research report for us. All the information in the report is factual and up-to-date.\n",
        "\n",
        "Note that this tutorial is meant to showcase the capability of the new search tool and inspire interesting use cases, not to build a production research report generator. If you are looking to use a tool, please check out Google [Deep Research](https://blog.google/products/gemini/google-gemini-deep-research/) in the Gemini app."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4677dd58e9b5"
      },
      "source": [
        "## Next Steps\n",
        "\n",
        "* To learn more about the search tool, check out the [Search tool documentation](https://ai.google.dev/gemini-api/docs/models/gemini-v2#search-tool).\n",
        "* To get started with streaming, check out the [Streaming Quickstart](../quickstarts/Streaming.ipynb).\n",
        "* To get started with the search tool, check out the [Search tool quick start](../quickstarts/Search_Grounding.ipynb).\n",
        "* To see how much easier and more powerful the search tool in Gemini 2.0 is than the previous search grounding feature, compare this tutorial with the [previous example](../examples/Search_grounding_for_research_report.ipynb).\n",
        "* To learn more about all the new Gemini 2.0 features, check out the [Gemini 2.0 quickstart notebooks](https://github.com/google-gemini/cookbook/tree/main/gemini-2/)."
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "Search_grounding_for_research_report.ipynb",
      "toc_visible": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}