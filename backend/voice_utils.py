import streamlit.components.v1 as components   # used to integrate custom HTML, CSS, JS into Streamlit

# embeds browser-based text-to-speech (TTS) controls directly into Streamlit 
def render_speech_buttons(response_text: str = "", lang: str = "en-US"):  
    """
    Render a client-side speech synthesis (read aloud) and stop button
    using the browser's Web Speech API (icon-only buttons).
    """
    escaped_text = (
        response_text.replace('"', '&quot;')
        .replace("\n", "\\n")
        .replace("'", "\\'")
    )

    html_code = f"""
    <div style="display:flex; align-items:center; gap:12px; margin-top:8px;">
        <!-- Speak Button -->
        <button id="speakBtn" title="Speak" style="
            border:none;
            background-color:#4CAF50;
            color:white;
            width:38px;
            height:38px;
            border-radius:50%;
            cursor:pointer;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:18px;
            box-shadow:0 2px 5px rgba(0,0,0,0.2);
        ">
            🔈
        </button>

        <!-- Stop Button -->
        <button id="stopBtn" title="Stop" style="
            border:none;
            background-color:#e53935;
            color:white;
            width:38px;
            height:38px;
            border-radius:50%;
            cursor:pointer;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:18px;
            box-shadow:0 2px 5px rgba(0,0,0,0.2);
        ">
            ⏹
        </button>
    </div>

    <script>
        const speakBtn = document.getElementById("speakBtn");
        const stopBtn = document.getElementById("stopBtn");
        const text = '{escaped_text}';

        function supportsSpeech() {{
            return 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined';
        }}

        speakBtn.addEventListener('click', () => {{
            if (!supportsSpeech()) {{
                alert("Your browser does not support text-to-speech.");
                return;
            }}
            window.speechSynthesis.cancel(); 
            const utter = new SpeechSynthesisUtterance(text);
            utter.lang = "{lang}";
            utter.rate = 1.0;
            utter.pitch = 1.0;
            const voices = window.speechSynthesis.getVoices();
            const matched = voices.find(v => v.lang.startsWith("{lang.split('-')[0]}"));
            if (matched) utter.voice = matched;
            window.speechSynthesis.speak(utter);
        }});

        stopBtn.addEventListener('click', () => {{
            window.speechSynthesis.cancel();
        }});
    </script>
    """
    components.html(html_code, height=60)
