from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the compiled model safely
MODEL_PATH = 'fake_profile_model.pkl'
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route('/')
def home():
    return render_template('index.html', evaluation_status=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', evaluation_status="System Error: Inference file missing.")

    try:
        # 1. Harvest user inputs from your HTML form fields
        followers = float(request.form['followers'])
        following = float(request.form['following'])
        posts = float(request.form['posts'])
        digits = float(request.form['digits'])
        pic = float(request.form['pic'])
        bio = float(request.form['bio'])
        
        # 2. Add baseline assumptions for the hidden dataset columns
        fullname_words = 2.0         # Assumed average word count for full name
        fullname_digit_ratio = 0.0   # Assumed clear of numbers
        name_equals_username = 0.0   # Assumed they are different
        is_private = 0.0             # Assumed public profile
        
        # 3. Apply your advanced mathematical transformations
        follower_following_ratio = followers / (following + 1)
        engagement_score = posts / (followers + 1)
        
        # 4. Construct the exact 12-feature array matching train.py order
        input_vector = np.array([[
            pic,                      # has_profile_pic
            digits,                   # username_digit_ratio
            fullname_words,           # fullname_words
            fullname_digit_ratio,     # fullname_digit_ratio
            name_equals_username,     # name_equals_username
            bio,                      # bio_length
            is_private,               # is_private
            posts,                    # post_count
            followers,                # followers_count
            following,                # following_count
            follower_following_ratio, # follower_following_ratio
            engagement_score          # engagement_score
        ]])
        
        # 5. Run classification prediction evaluation
        prediction = model.predict(input_vector)
        
        if prediction == 1:
            status = "⚠️ FAKE PROFILE ALERT: Structural anomaly patterns detected."
            css = "danger-alert"
        else:
            status = "✅ GENUINE ACCOUNT VERIFIED: Metrics represent standard human profiles."
            css = "success-alert"
            
        return render_template('index.html', evaluation_status=status, alert_style=css)
        
    except Exception as e:
        return render_template('index.html', evaluation_status=f"Parsing Exception: {str(e)}", alert_style="danger-alert")

if __name__ == '__main__':
    app.run(debug=True)
