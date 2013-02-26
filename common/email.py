'''
Created on Feb 20, 2013

@author: guoguibing
'''
import smtplib, os

from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from email.mime.base import MIMEBase
from email import encoders

def send_email(From='guobingyou@gmail.com', To='gguo1@e.ntu.edu.sg'):
    host = 'mail.gmail.com'
    
    msg = MIMEMultipart()
    msg['From'] = From
    msg['To'] = To
    msg['Subject'] = 'Program is finished.'
    msg['Date'] = formatdate(localtime=True)
    
    # attach a file
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(open('cf.config', 'rb').read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"'% os.path.basename('cf.config'))
    msg.attach(part)
    
    server = smtplib.SMTP(host)
    server.login('guobingyou', 'through@pass')
    
    try:
        server.sendmail(From, To, msg.as_string())
        server.close()
    except Exception, e:
        error_msg = 'Unable to send email. Error: %s' % str(e)
        print error_msg

if __name__=='__main__':
    send_email()