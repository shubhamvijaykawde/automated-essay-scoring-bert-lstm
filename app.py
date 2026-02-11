from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer
from bert_models.bert_regressor import BertMultiOutputRegressor

app = Flask(__name__)

device = torch.device("cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertMultiOutputRegressor(num_outputs=6)
model.load_state_dict(torch.load("models/bert_cela_regressor.pt", map_location=device))
model.eval()

# Labels, max scores, and whether to calibrate
PARAMETERS = [
    ("Grammar", 8, True),
    ("Lexical", 8, True),
    ("Global Organization", 8, True),
    ("Local Organization", 8, True),
    ("Supporting Ideas", 8, True),
    ("Holistic", 5, False),  # No calibration for holistic
]

# Smart calibration function
def calibrate(score, max_score):
    """
    Calibration rules:
    - Bad essays (score <= 4) are NOT boosted
    - Moderate essays (4 < score < 0.7*max_score) get a small boost
    - High essays (>=0.7*max_score) remain unchanged
    """
    if score <= 3.5:
        return score
    elif score < max_score * 0.7:
        return min(max_score, score * 1.1)  # mild boost
    else:
        return min(max_score, score)  # cap at max_score

# Feedback rules
def generate_feedback(scores):
    feedback = {}
    # ---------- Grammar (/8) ----------
    g = scores["Grammar"]
    if g < 2:
        feedback["Grammar"] = "Frequent grammatical errors severely affect clarity and readability."
    elif g < 4:
        feedback["Grammar"] = "Noticeable grammatical errors are present and reduce overall clarity."
    elif g < 6:
        feedback["Grammar"] = "Some grammatical issues exist, but meaning remains generally clear."
    else:
        feedback["Grammar"] = "Grammar is largely accurate with only minor errors."

    # ---------- Lexical (/8) ----------
    l = scores["Lexical"]
    if l < 2:
        feedback["Lexical"] = "Very limited vocabulary with repetitive and inappropriate word choices."
    elif l < 4:
        feedback["Lexical"] = "Basic vocabulary is used, but expression lacks variety and precision."
    elif l < 6:
        feedback["Lexical"] = "Adequate vocabulary range, though more variation would improve expression."
    else:
        feedback["Lexical"] = "Rich and appropriate vocabulary enhances the quality of the essay."

    # ---------- Global Organization (/8) ----------
    go = scores["Global Organization"]
    if go < 2:
        feedback["Global Organization"] = "The essay lacks a clear structure and logical progression."
    elif go < 4:
        feedback["Global Organization"] = "Organization is weak, with unclear transitions between ideas."
    elif go < 6:
        feedback["Global Organization"] = "Overall structure is present, but coherence can be improved."
    else:
        feedback["Global Organization"] = "The essay is well-structured and logically organized."

    # ---------- Local Organization (/8) ----------
    lo = scores["Local Organization"]
    if lo < 2:
        feedback["Local Organization"] = "Sentences and paragraphs are poorly connected, disrupting flow."
    elif lo < 4:
        feedback["Local Organization"] = "Limited cohesion between sentences makes reading difficult at times."
    elif lo < 6:
        feedback["Local Organization"] = "Generally coherent, though some transitions could be smoother."
    else:
        feedback["Local Organization"] = "Strong sentence flow and paragraph cohesion throughout."

    # ---------- Supporting Ideas (/8) ----------
    si = scores["Supporting Ideas"]
    if si < 2:
        feedback["Supporting Ideas"] = "Arguments are unclear, insufficient, or largely unsupported."
    elif si < 4:
        feedback["Supporting Ideas"] = "Some supporting ideas are present, but lack clarity or relevance."
    elif si < 6:
        feedback["Supporting Ideas"] = "Relevant arguments are provided, though depth could be improved."
    else:
        feedback["Supporting Ideas"] = "Ideas are well-supported with clear and relevant arguments."

    # ---------- Holistic (/5) ----------
    h = scores["Holistic"]
    if h < 1.5:
        feedback["Holistic"] = "Overall essay quality is weak and requires substantial improvement."
    elif h < 2.5:
        feedback["Holistic"] = "The essay demonstrates basic competence but lacks overall effectiveness."
    elif h < 3.5:
        feedback["Holistic"] = "Adequate overall quality, though several areas need refinement."
    else:
        feedback["Holistic"] = "Strong overall essay quality with effective communication."

    return feedback


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        essay = request.form["essay"]

        # Tokenize essay
        inputs = tokenizer(
            essay,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # Predict raw scores
        with torch.no_grad():
            raw_scores = model(
                inputs["input_ids"],
                inputs["attention_mask"]
            ).squeeze().tolist()

        # Map scores to parameters and apply selective calibration
        results = {}
        for (label, max_score, apply_calib), value in zip(PARAMETERS, raw_scores):
            if apply_calib:
                calibrated = calibrate(float(value), max_score)
            else:
                calibrated = float(value)  # holistic untouched
            results[label] = {
                "score": round(calibrated, 2),
                "max": max_score
            }

        numeric_scores = {k: v["score"] for k, v in results.items()}
        feedback = generate_feedback(numeric_scores)

        return render_template(
            "index.html",
            essay=essay,
            results=results,
            feedback=feedback
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
