# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.message import EmailMessage

msg = EmailMessage()
msg.set_content('nothing new in the west')

# me == the sender's email address
# you == the recipient's email address
msg['Subject'] = 'Does Google block this?'
msg['From'] = 'secret@stanford.edu'
msg['To'] = 'thaifunn92@gmail.com'

# Send the message via our own SMTP server.
s = smtplib.SMTP('localhost')
s.send_message(msg)
s.quit()