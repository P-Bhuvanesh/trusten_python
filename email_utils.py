from email.message import EmailMessage
import os
import logging
import datetime
from aiosmtplib import SMTP
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

load_dotenv()

# Configure logging specifically for email operations
mail_logger = logging.getLogger('email_service')
mail_logger.setLevel(logging.INFO)

# Create file handler for mail logs
mail_handler = logging.FileHandler('mail_logs.log')
mail_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
mail_logger.addHandler(mail_handler)

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
APP_NAME = "Trusten.Vision"

async def send_email_async(to_email: str, subject: str, body: str):
    mail_logger.info(f"Preparing to send email to {to_email} with subject: {subject}")
    
    # Use MIMEMultipart for HTML emails
    message = MIMEMultipart("alternative")
    message["From"] = f"{APP_NAME} <{GMAIL_USER}>"
    message["To"] = to_email
    message["Subject"] = subject
    message["Reply-To"] = GMAIL_USER
    message.add_header("List-Unsubscribe", f"<mailto:{GMAIL_USER}?subject=unsubscribe>")
    
    # Plain text version
    text_part = MIMEText(body, "plain")
    
    # HTML version with professional formatting
    body_html = body.replace("\n", "<br>")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{subject}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #0066cc; padding: 10px; color: white; text-align: center; }}
            .content {{ padding: 20px; background-color: #ffffff; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>{APP_NAME}</h2>
            </div>
            <div class="content">
                {body_html}
            </div>
            <div class="footer">
                <p>This is an automated message from {APP_NAME}. Please do not reply to this email.</p>
                <p>If you believe you received this email in error, please contact support.</p>
                <p>&copy; {APP_NAME} {datetime.datetime.now().year}</p>
            </div>
        </div>
    </body>
    </html>
    """
    html_part = MIMEText(html_content, "html")
    
    # Attach both plain and HTML parts
    message.attach(text_part)
    message.attach(html_part)

    smtp = None
    try:
        mail_logger.info("Connecting to SMTP server with STARTTLS...")
        smtp = SMTP(
            hostname="smtp.gmail.com",
            port=587,
            timeout=30,
            start_tls=True 
        )
        await smtp.connect()
        mail_logger.info("Connection established with TLS, attempting login...")
        await smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        mail_logger.info("Login successful, sending message...")
        await smtp.send_message(message)
        mail_logger.info(f"Email successfully sent to {to_email}")
        return {"status": "success"}
    except Exception as e:
        error_message = f"Failed to send email: {str(e)}"
        mail_logger.error(error_message)
        return {"status": "error", "message": error_message}
    finally:
        if smtp is not None:
            try:
                if hasattr(smtp, 'is_connected') and smtp.is_connected:
                    mail_logger.info("Closing SMTP connection...")
                    await smtp.quit()
                    mail_logger.info("SMTP connection closed properly")
            except Exception as e:
                mail_logger.warning(f"Error during SMTP connection cleanup: {str(e)}")
