import smtplib
from email.mime.text import MIMEText
def mail(emotion_voice,name, email):
    # define content
    recipients = email
    sender = "kidnapalert@gmail.com"
    subject = "Kidnap Alert"
    body = """
    Dear people,
    """ + name + """Is in a kidnaped so please try to contact
    Voice Emotion is :-"""+emotion_voice

    # make up message
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)

    # sending
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender, 'alertkidnap')
    send_it = session.sendmail(sender, recipients, msg.as_string())
    session.quit()
