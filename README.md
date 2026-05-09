# curly-train
Generative AI model built to predict Anime Dialogue(Class Room of The Elite)

## Web App

Backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend talks to the API at `http://127.0.0.1:8000` by default. Set `VITE_API_URL` if you want a different backend address.
