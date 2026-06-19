import os
import json
import cv2
import base64
from openai import OpenAI

# -------------------------
# Configuration
# -------------------------
### TODO: Ensure OPENAI_API_KEY is set in your environment variables.
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = "qwen3.6-35b-a3b"

IMAGE_PATH = "test.jpg"
OUTPUT_PATH = "grounding_result.jpg"
QUERY = "Locate the dog. Report bbox coordinates in JSON format."

def encode_image(image_path: str) -> str:
    """Read an image and encode it to base64."""
    ### TODO: Read the image file and encode it as a base64 string.
    ### Decode it to "utf-8" to get a clean string.
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')

    return base64_string

def call_vlm_grounding(client: OpenAI, base64_image: str, query: str) -> str:
    """Call the VLM API with the image and query."""
    ### TODO: Construct the messages for the chat completion.
    ### Provide the base64 image (data:image/jpeg;base64,{base64_image}) and the QUERY text.
    ### Call client.chat.completions.create with the correct model and messages.
    ### Return the text content of the response.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": query
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    return response.choices[0].message.content

def parse_bbox_from_response(response: str) -> tuple[int, int, int, int]:
    """Parse the [x1, y1, x2, y2] bounding box from the VLM JSON response."""
    ### TODO: The response should contain a JSON block (e.g., ```json ... ```).
    ### Parse the JSON to extract the "bbox_2d" coordinates [x1, y1, x2, y2].
    ### Handle potential markdown formatting.
    import re
    match = re.search(r"```(?:json)?\n?(.*?)\n?```", response, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
    else:
        json_str = response.strip()

    data = json.loads(json_str)

    if isinstance(data, list) and len(data) > 0:
        bbox = data[0].get("bbox_2d", [0, 0, 0, 0])
    elif isinstance(data, dict):
        bbox = data.get("bbox_2d", [0, 0, 0, 0])
    else:
        bbox = [0, 0, 0, 0]

    return tuple(bbox)

def visualize_and_save(image_path: str, bbox: tuple[int, int, int, int], output_path: str):
    """Denormalize coordinates and draw the bounding box on the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path} for visualization.")
        
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    ### TODO: Denormalize the bounding box coordinates.
    ### Qwen3-VL API outputs relative coordinates [0, 1000].
    ### Convert them back to the original image pixel dimensions (h, w).
    abs_x1, abs_y1, abs_x2, abs_y2 = int((x1 / 1000.0) * w), int((y1 / 1000.0) * h), int((x2 / 1000.0) * w), int((y2 / 1000.0) * h)

    cv2.rectangle(
        img,
        (abs_x1, abs_y1),
        (abs_x2, abs_y2),
        (0, 255, 0),
        2,
    )

    cv2.imwrite(output_path, img)
    print(f"Saved visualization -> {output_path}")

def main():
    print("1. Initializing OpenAI client for DashScope...")
    if not API_KEY:
        print("Error: API_KEY is not set. Please set the DASHSCOPE_API_KEY environment variable.")
        return
        
    ### TODO: Initialize the OpenAI client with your API_KEY and BASE_URL.
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    try:
        print(f"2. Reading and encoding image {IMAGE_PATH}...")
        base64_image = encode_image(IMAGE_PATH)

        print("3. Generating response via API...")
        response = call_vlm_grounding(client, base64_image, QUERY)
        print("Model Output:\n", response)

        print("4. Parsing Bounding Box...")
        bbox = parse_bbox_from_response(response)

        print("5. Visualizing result...")
        visualize_and_save(IMAGE_PATH, bbox, OUTPUT_PATH)

    except Exception as e:
        print(f"Error during grounding task: {e}")

if __name__ == "__main__":
    main()
