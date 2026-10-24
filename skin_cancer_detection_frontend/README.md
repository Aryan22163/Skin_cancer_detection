# DermaScan front end

## Files
- `index.html` — page structure
- `style.css` — responsive styling
- `script.js` — image selection, preview, and API request

## Important: connect your trained model
This is the front end only. It expects a backend endpoint at `POST /api/predict` that accepts a multipart form field named `image` and returns JSON like:

```json
{"prediction": "class name", "confidence": 0.93}
```

`confidence` can be 0–1 or 0–100. Do not present the result as a medical diagnosis.

## Run locally
Serve this folder using a local web server (rather than opening `index.html` directly), and configure `API_ENDPOINT` in `script.js` if your backend uses another URL. For deployment, host the front end and backend together or set the endpoint to your deployed API URL. If front end and API use different domains, configure CORS on the backend.

This is an educational demo, not a clinical tool. AI results can be wrong; medical concerns should be assessed by a qualified dermatologist.
