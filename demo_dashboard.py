
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    # Examples designed to trigger the ML model based on its features:
    # 1. Safe: Standard domain, https
    # 2. Phishing: Homoglyph (paypa1), suspicious TLD (.xyz), keywords (secure-login)
    # 3. Malware: IP address, suspicious extension (.exe)
    # 4. Scam: Random looking domain, multiple subdomains
    
    test_cases = [
        {
            "label": "✅ SAFE CONTRACT",
            "url": "https://legit-verify.com/verify/contract-001",
            "desc": "Standard domain with HTTPS. Should be ALLOWED."
        },
        {
            "label": "❌ PHISHING ATTEMPT",
            "url": "http://paypa1-secure-login.xyz/verify/contract-999",
            "desc": "Triggers: Homoglyph (paypa1), Suspicious TLD (.xyz), Keywords (secure, login), No HTTPS."
        },
        {
            "label": "❌ MALWARE LINK",
            "url": "http://192.168.1.45/verify/update_checker.exe",
            "desc": "Triggers: IP Address as domain, Suspicious file extension (.exe)."
        },
        {
            "label": "❌ SCAM SITE",
            "url": "http://win-free-prize.secure-verify.tk/verify/claim",
            "desc": "Triggers: Misleading TLD (.tk), keywords (win, free, secure), excessive subdomains."
        }
    ]
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ShieldNet ML Demo</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; margin: 0; padding: 40px; text-align: center; }
            .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
            .card { background: white; border-radius: 12px; padding: 20px; width: 300px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s; }
            .card:hover { transform: translateY(-5px); }
            .qr { width: 200px; height: 200px; margin-bottom: 15px; }
            h1 { color: #1a73e8; margin-bottom: 30px; }
            h3 { margin: 10px 0; font-size: 1.1em; }
            p { font-size: 0.85em; color: #666; height: 40px; }
            .url { font-family: monospace; font-size: 0.75em; background: #eee; padding: 5px; word-break: break-all; border-radius: 4px; }
            .safe { color: #28a745; }
            .danger { color: #dc3545; }
        </style>
    </head>
    <body>
        <h1>ShieldNet Threat Detection Demo</h1>
        <p>Scan these QR codes with the <b>Vaultkey Mobile App</b> to test the ML blocking logic.</p>
        
        <div class="container">
    """
    
    for case in test_cases:
        color_class = "safe" if "SAFE" in case["label"] else "danger"
        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={case['url']}"
        
        html_content += f"""
            <div class="card">
                <h3 class="{color_class}">{case['label']}</h3>
                <img class="qr" src="{qr_url}" alt="QR Code">
                <p>{case['desc']}</p>
                <div class="url">{case['url']}</div>
            </div>
        """
        
    html_content += """
        </div>
        <div style="margin-top: 50px; color: #888; font-size: 0.8em;">
            Running ShieldNet v2 ML Model on-device (Android TFLite)
        </div>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    print("\n🚀 Starting Demo Dashboard on http://localhost:8000")
    print("Scan the QR codes from your mobile device to test the blocking logic.\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
