# 🎥 AI Video Assistant — Video Summarization & RAG Chat

An end-to-end AI pipeline that turns any video, meeting recording, or audio file into structured, 
searchable knowledge. Simply provide a YouTube URL or a local video/audio file, and the system 
automatically transcribes it, generates a concise summary, extracts key action items and decisions, 
and builds a Retrieval-Augmented Generation (RAG) chat interface — so you can ask questions directly 
about the content instead of re-watching it.

## ✨ Features
- 🎙️ **Automatic Transcription** — Converts speech from video/audio into accurate text (supports English & Hinglish).
- 📋 **Smart Summarization** — Generates a clear, concise summary and an auto-generated title for the content.
- ✅ **Action Item Extraction** — Identifies tasks and follow-ups discussed in the video.
- 🔑 **Key Decision Detection** — Highlights important decisions made during the conversation.
- ❓ **Open Question Mining** — Surfaces unresolved questions for quick follow-up.
- 💬 **Chat with Your Video (RAG)** — Ask natural-language questions about the transcript and get context-aware answers powered by a vector-based RAG engine.
- 🖥️ **Interactive Streamlit UI** — Clean, professional interface to upload sources, track processing stages, and explore results in real time.

## 🧠 Tech Stack
- **Python**
- **LangChain** (RAG pipeline, summarization, extraction)
- **Streamlit** (interactive web UI)
- **Vector Store** for semantic search over transcripts
- **Whisper / Speech-to-Text** for transcription

## 🚀 How It Works
1. Provide a YouTube link or local video/audio file.
2. The pipeline processes and transcribes the audio.
3. AI generates a title, summary, action items, decisions, and open questions.
4. A RAG-based chat index is built so you can interactively ask questions about the video.

## 📦 Use Cases
- Summarizing long meetings, lectures, or interviews
- Extracting actionable tasks from recorded calls
- Quickly searching through long video content without rewatching
