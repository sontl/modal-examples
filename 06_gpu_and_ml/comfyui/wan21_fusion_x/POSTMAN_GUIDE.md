# Postman Testing Guide for WanFusionX Image-to-Video API

This guide shows you how to test the image-to-video endpoints using Postman.

## Endpoint 1: Form-Data Endpoint (Recommended for Postman)

**URL:** `https://your-modal-app-url/image-to-video`

### Steps:
1. Set method to **POST**
2. In the **Body** tab, select **form-data**
3. Add two fields:
   - **Key:** `image` | **Type:** File | **Value:** Select your image file
   - **Key:** `prompt` | **Type:** Text | **Value:** Your custom prompt (optional)

### Example:
```
Key: image     | Type: File | Value: [Select your image file]
Key: prompt    | Type: Text | Value: "The person's eyes glow with magical energy"
```

## API Documentation

The API now uses a single, well-structured endpoint that supports both image upload and custom prompts through form-data.

## Expected Response

Both endpoints will return:
- **Status:** 200 OK
- **Content-Type:** `video/mp4`
- **Body:** Binary video data

## Saving the Response

1. After sending the request, click **Save Response**
2. Choose **Save to file**
3. Save with `.mp4` extension (e.g., `generated_video.mp4`)

## Troubleshooting

### Common Errors:

1. **"Field required" error:**
   - Make sure you're using the correct endpoint
   - For form-data endpoint, ensure the field name is exactly `image`

2. **Timeout errors:**
   - Video generation can take 2-5 minutes
   - Increase Postman timeout in Settings > General > Request timeout

3. **File size errors:**
   - Keep image files under 10MB
   - Supported formats: JPG, PNG, GIF, BMP

### Tips:
- The endpoint supports both image upload and custom prompts
- Video generation typically takes 2-5 minutes depending on image complexity
- You can access the interactive API documentation at `https://your-modal-app-url/docs`