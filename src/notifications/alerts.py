"""
Trading alerts via WhatsApp and Email
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import os
from datetime import datetime


class AlertSystem:
    """Send trading alerts via multiple channels"""
    
    def __init__(self):
        # Email configuration (using Gmail)
        self.email_enabled = os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true'
        self.email_from = os.getenv('EMAIL_FROM', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        self.email_to = os.getenv('EMAIL_TO', '')
        
        # WhatsApp configuration (using Twilio or similar service)
        self.whatsapp_enabled = os.getenv('WHATSAPP_ALERTS_ENABLED', 'false').lower() == 'true'
        self.whatsapp_api_key = os.getenv('WHATSAPP_API_KEY', '')
        self.whatsapp_number = os.getenv('WHATSAPP_NUMBER', '')
    
    def send_email(self, subject, body, html=False):
        """Send email alert"""
        
        if not self.email_enabled or not all([self.email_from, self.email_password, self.email_to]):
            print("⚠️  Email not configured. Skipping email alert.")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = f"PSX Alert: {subject}"
            
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Gmail SMTP
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email_from, self.email_password)
            server.send_message(msg)
            server.quit()
            
            print(f"✓ Email sent: {subject}")
            return True
            
        except Exception as e:
            print(f"✗ Email failed: {e}")
            return False
    
    def send_whatsapp(self, message):
        """Send WhatsApp alert (using Twilio)"""
        
        if not self.whatsapp_enabled or not all([self.whatsapp_api_key, self.whatsapp_number]):
            print("⚠️  WhatsApp not configured. Skipping WhatsApp alert.")
            return False
        
        try:
            # Using Twilio WhatsApp API
            from twilio.rest import Client
            
            account_sid = os.getenv('TWILIO_ACCOUNT_SID', '')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN', '')
            
            if not all([account_sid, auth_token]):
                print("⚠️  Twilio not configured properly.")
                return False
            
            client = Client(account_sid, auth_token)
            
            msg = client.messages.create(
                from_=f'whatsapp:{self.whatsapp_api_key}',
                body=message,
                to=f'whatsapp:{self.whatsapp_number}'
            )
            
            print(f"✓ WhatsApp sent: {msg.sid}")
            return True
            
        except Exception as e:
            print(f"✗ WhatsApp failed: {e}")
            return False
    
    def send_trading_signal(self, signal):
        """Send trading signal alert"""
        
        symbol = signal['symbol']
        recommendation = signal['recommendation']
        price = signal['current_price']
        predicted = signal['predicted_price']
        change = signal['predicted_change_pct']
        confidence = signal['confidence']
        
        # Determine emoji
        emoji = '🟢' if recommendation == 'BUY' else '🔴' if recommendation == 'SELL' else '🟡'
        
        # Email body (HTML)
        email_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {'#28a745' if recommendation == 'BUY' else '#dc3545' if recommendation == 'SELL' else '#ffc107'};">
                {emoji} {recommendation} Signal: {symbol}
            </h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Current Price:</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">PKR {price:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Predicted Price:</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">PKR {predicted:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Expected Change:</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd; color: {'green' if change > 0 else 'red'};">{change:+.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Confidence:</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{confidence:.1%}</td>
                </tr>
            </table>
            <p style="margin-top: 20px; color: #666;">
                Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """
        
        # WhatsApp message (Plain text)
        whatsapp_msg = f"""
{emoji} *{recommendation} SIGNAL*

*Symbol:* {symbol}
*Current:* PKR {price:.2f}
*Predicted:* PKR {predicted:.2f}
*Change:* {change:+.2f}%
*Confidence:* {confidence:.1%}

_{datetime.now().strftime('%Y-%m-%d %H:%M')}_
        """
        
        # Send alerts
        self.send_email(f"{recommendation} - {symbol}", email_body, html=True)
        self.send_whatsapp(whatsapp_msg)
    
    def send_daily_summary(self, summary_data):
        """Send daily market summary"""
        
        subject = f"Daily PSX Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Email HTML
        email_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>PSX Daily Summary</h2>
            <h3>Top Opportunities</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px; border: 1px solid #ddd;">Symbol</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">Signal</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">Change</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">Confidence</th>
                </tr>
        """
        
        for signal in summary_data.get('top_signals', [])[:10]:
            color = '#28a745' if signal['recommendation'] == 'BUY' else '#dc3545'
            email_body += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">{signal['symbol']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; color: {color};">{signal['recommendation']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{signal['predicted_change_pct']:+.2f}%</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{signal['confidence']:.1%}</td>
                </tr>
            """
        
        email_body += """
            </table>
        </body>
        </html>
        """
        
        self.send_email(subject, email_body, html=True)


# Setup configuration
def setup_alerts():
    """Guide user through alert setup"""
    
    print("\n" + "="*70)
    print("ALERT SYSTEM CONFIGURATION")
    print("="*70 + "\n")
    
    env_content = """
# Email Alerts Configuration
EMAIL_ALERTS_ENABLED=true
EMAIL_FROM=gulfayazrumi5@gmail.com
EMAIL_PASSWORD=your_app_password  # Use Gmail App Password
EMAIL_TO=recipient@email.com

# WhatsApp Alerts Configuration (Twilio)
WHATSAPP_ALERTS_ENABLED=false
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
WHATSAPP_API_KEY=+14155238886  # Twilio WhatsApp number
WHATSAPP_NUMBER=+92XXXXXXXXXX  # Your number with country code
"""
    
    print("Add these to your .env file:")
    print("-"*70)
    print(env_content)
    print("-"*70)
    
    print("\n📧 Email Setup (Gmail):")
    print("  1. Go to Google Account settings")
    print("  2. Enable 2-Factor Authentication")
    print("  3. Generate App Password: https://myaccount.google.com/apppasswords")
    print("  4. Use that app password in EMAIL_PASSWORD")
    
    print("\n📱 WhatsApp Setup (Twilio):")
    print("  1. Create Twilio account: https://www.twilio.com/")
    print("  2. Get WhatsApp sandbox number")
    print("  3. Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")
    print("  4. Send 'join <sandbox-code>' to Twilio WhatsApp number")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    setup_alerts()