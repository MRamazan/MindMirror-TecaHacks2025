from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from llama_cpp import Llama
import os
from datetime import timedelta

app = Flask(__name__)


app.secret_key = '123'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)


CORS(app,
     origins=["http://localhost:5000", "http://127.0.0.1:5000"],
     supports_credentials=True,
     allow_headers=["Content-Type"])


llm = None


@app.route('/')
def index():
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404


def load_model():
    global llm
    try:
        print("Model loading...")
        llm = Llama.from_pretrained(
            repo_id="Kush26/Mental_Health_ChatBot",
            filename="all_files/unsloth.Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Model loading error: {str(e)}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    model_status = "loaded" if llm is not None else "not loaded"
    return jsonify({
        "status": "running",
        "model_status": model_status
    })

ARTICLES = [
        {
            "name": "Texas Tech University",
            "link": "https://www.depts.ttu.edu/rise/Blog/midsemesterslump.php"
        },
        {
            "name": "George Fox University",
            "link": "https://blogs.georgefox.edu/dlgp/screened-lives-kehidupan-yang-disaring/"
        },
{
            "name": "NC State Extension",
            "link": "https://news.ces.ncsu.edu/digital-detox-finding-the-balance-in-a-hyper-connected-world/"
        },
        {
            "name": "Boston University",
            "link": "https://www.bu.edu/articles/2023/social-media-adolescent-health-report/"
        },
        {
            "name": "University of Utah Health",
            "link": "https://healthcare.utah.edu/healthfeed/2023/01/impact-of-social-media-teens-mental-health"
        },
        {
            "name": "University of Utah Health",
            "link": "https://healthcare.utah.edu/the-scope/kids-zone/all/2024/10/social-media-taking-over-your-teens-life-what-you-can-do-about-it"
        },
        {
            "name": "University of Texas Permian Basin",
            "link": "https://online.utpb.edu/about-us/articles/psychology/thriving-in-the-digital-age-how-technology-influences-our-behavior/"
        },
        {
            "name": "Ohio State University",
            "link": "https://u.osu.edu/emotionalfitness/?p=621"
        },
        {
            "name": "Penn State University",
            "link": "https://sites.psu.edu/aspsy/2024/10/23/tick-tock/"
        },
        {
            "name": "University of Richmond",
            "link": "https://jolt.richmond.edu/2024/03/06/tiktok-brain-can-we-save-childrens-attention-spans/"
        },
        {
            "name": "George Mason University",
            "link": "https://graduate.gmu.edu/news/2022-11/why-am-i-tired-because-i-am-tired"
        },
        {
            "name": "Mercy University",
            "link": "https://career.mercy.edu/blog/2025/02/24/8-ways-to-spot-burnout-before-it-derails-your-career/"
        },
        {
            "name": "University of Massachusetts Boston",
            "link": "https://blogs.umb.edu/undercurrents/2025/08/18/the-dystopian-landscape-of-short-form-content/"
        },
        {
            "name": "Harvard Health Publishing",
            "link": "https://www.health.harvard.edu/blog/staying-focused-in-the-era-of-digital-distractions-2020060920152"
        },
        {
            "name": "Harvard Summer School",
            "link": "https://summer.harvard.edu/blog/need-a-break-from-social-media-heres-why-you-should/"
        },
        {
            "name": "UC Davis Health",
            "link": "https://health.ucdavis.edu/blog/cultivating-health/social-medias-impact-on-our-mental-health/2024/05"
        },

{
            "name": "Holy Family University",
            "link": "https://www.holyfamily.edu/about/news-and-media/hfu-blog-network/tiktok-impact-attention-and-memory"
        },
        {
            "name": "Holy Family University",
            "link": "https://www.holyfamily.edu/about/news-and-media/hfu-blog-network/cognitive-and-emotional-consequences-hurry-sickness-how-constant-rushing-overloading-your-brain"
        },
]

@app.route('/articles', methods=['GET'])
def get_articles():
    try:
        return jsonify({
            "status": "success",
            "count": len(ARTICLES),
            "articles": ARTICLES
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if llm is None:
            if not load_model():
                return jsonify({
                    "error": "Model could not be loaded. Please check the logs."
                }), 500

        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({
                "error": "Message cannot be empty"
            }), 400

        if 'conversation_history' not in session:
            session['conversation_history'] = []
            session.permanent = True

        session['conversation_history'].append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {
                "role": "system",
                "content": "You are a compassionate mental health assistant. Provide supportive, empathetic responses about mental wellness, brain health, and digital wellbeing. Keep your responses concise and helpful."
            }
        ]

        messages.extend(session['conversation_history'][-10:])

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        bot_response = response["choices"][0]["message"]["content"]

        session['conversation_history'].append({
            "role": "assistant",
            "content": bot_response
        })

        session.modified = True

        return jsonify({
            "response": bot_response,
            "status": "success"
        })

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "response": "I'm sorry, I encountered an error. Please try again."
        }), 500


@app.route('/reset', methods=['POST'])
def reset_conversation():
    try:
        session.pop('conversation_history', None)
        session.modified = True

        return jsonify({
            "status": "success",
            "message": "Conversation history cleared successfully"
        })
    except Exception as e:
        print(f"Reset error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Could not reset conversation: {str(e)}"
        }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("MindMirror Mental Health Chatbot Backend")
    print("=" * 50)

    if load_model():
        print("\n✅ Backend ready!")
        print("🌐 Frontend: http://localhost:5000")
        print("💬 Chat endpoint: http://localhost:5000/chat")
        print("🔄 Reset endpoint: http://localhost:5000/reset")
        print("=" * 50)
    else:
        print("\n⚠️ Model could not be loaded, but server is starting...")
        print("Model will be loaded on first chat request")
        print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )