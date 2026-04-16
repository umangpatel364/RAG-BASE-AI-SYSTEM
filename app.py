from flask import Flask, render_template, request, jsonify
from rag_utils import rag_pipeline

app = Flask(__name__)

# ---------------- HOME ---------------- #
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- CHAT API ---------------- #
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    try:
        # Directly call RAG pipeline
        final_response = rag_pipeline(user_message)
        return jsonify({"response": final_response})

    except Exception as e:
        # Show real error during debugging
        return jsonify({"response": str(e)})


# ---------------- VIDEOS PAGE ---------------- #
@app.route("/videos")
def videos_page():

    video_data = [
        ("1_Installing VS Code & How Websites Work", "tVzUXW6siu0"),
        ("2_Your First HTML Website", "kJEsTjH5mVg"),
        ("3_Basic Structure of HTML Website", "BGeDBfCIqas"),
        ("4_Heading, Paragraphs and Links", "nXba2-mgn1k"),
        ("5_Images, Lists, and Tables", "1BsVhumGlNc"),
        ("6_SEO and Core Web Vitals", "CyRlWlaJnTY"),
        ("7_Forms and Input Tags", "tLBlhp0SA_0"),
        ("8_Inline & Block Elements", "vnnlUCLfn6I"),
        ("9_Id & Classes in HTML", "vlAWzsGd-Yk"),
        ("10_Video, Audio & Media in HTML", "XZwBNDGuWGU"),
        ("11_Semantic Tags in HTML", "fhoDRB53DwY"),
        ("12_Exercise 1 - Pure HTML Media Player", "5xFRg_TzlAg"),
        ("13_Entities, Code tag and more on HTML", "cvsbHZcDx8w"),
        ("14_Introduction to CSS", "1dkfuga2_Ps"),
        ("15_Inline, Internal & External CSS", "-XwZpYIyCEA"),
        ("16_Exercise 1 - Solution & Shoutouts", "anGMeDGvZhw"),
        ("17_CSS Selectors MasterClass", "1cEG1T8beO4"),
        ("18_CSS Box Model - Margin, Padding & Borders", "Xrxd6cEajhM"),
    ]

    videos = [{"title": title, "id": vid} for title, vid in video_data]

    return render_template("videos.html", videos=videos)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)