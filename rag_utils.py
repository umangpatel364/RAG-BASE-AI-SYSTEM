import requests
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Convert seconds to minute-second format
def seconds_to_min_sec(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes} min {sec} sec"


# Load embeddings once
df = joblib.load("embeddings.joblib")


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        },
        timeout=60
    )
    return r.json()["embeddings"]


def generate_answer(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    return r.json()["response"]


def rag_pipeline(user_query):

    # 1️⃣ Create embedding
    question_embedding = create_embedding([user_query])[0]

    # 2️⃣ Compute similarity
    similarities = cosine_similarity(
        np.vstack(df["embedding"]),
        [question_embedding]
    ).flatten()

    # 🚨 Similarity Guard (Very Important)
    if similarities.max() < 0.30:
        return "This assistant answers only questions related to the Sigma Web Development course."

    # 3️⃣ Get top matches
    top_k = 5
    max_index = similarities.argsort()[::-1][:top_k]
    new_df = df.loc[max_index]

    if new_df.empty:
        return "This assistant answers only questions related to the Sigma Web Development course."

    # 4️⃣ Detect most relevant video
    video_number = new_df["number"].mode()[0]
    video_df = new_df[new_df["number"] == video_number]

    video_title = video_df.iloc[0]["title"]

    # 5️⃣ Get time range
    start_time = video_df["start"].min()
    end_time = video_df["end"].max()

    start_time_str = seconds_to_min_sec(start_time)
    end_time_str = seconds_to_min_sec(end_time)

    # 6️⃣ Create context
    context = video_df["text"].str.cat(sep=" ")

    # 7️⃣ Clean prompt
    prompt = f"""
You are a professional web development course assistant.

Answer clearly in 3–5 sentences.

Rules:
- Mention: Video {video_number}: "{video_title}"
- Do not include timestamps in explanation.
- Keep response clean and professional.
- No dramatic tone.
- No follow-up questions.

Context:
{context}

Question:
{user_query}
"""

    # 8️⃣ Generate answer
    answer = generate_answer(prompt)

    # 9️⃣ Final formatted response
    final_response = (
        answer.strip()
        + f"\n\nStart Time: {start_time_str}"
        + f"\nEnd Time: {end_time_str}"
        + "\n\nYou can explore this topic deeply from the COURSE section."
    )

    return final_response