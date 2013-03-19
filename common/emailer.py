'''
Created on Feb 20, 2013

@author: guoguibing
'''
import smtplib, os

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from email.mime.text import MIMEText

def send_email(**params):
    host = 'smtp.gmail.com'
    
    Subject = params['Subject'] if 'Subject' in params else 'Program is finished'
    From = params['From'] if 'From' in params else 'guobingyou@gmail.com'
    To = params['To'] if 'To' in params else 'gguo1@e.ntu.edu.sg'
    
    attch = params['file'] if 'file' in params else ''
    text = params['text'] if 'text' in params else 'some texts'
    
    msg = MIMEMultipart()
    msg['From'] = From
    msg['To'] = To
    msg['Subject'] = Subject
    msg['Date'] = formatdate(localtime=True)
    
    # attach a file
    if file != '':
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(open(attch, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(attch))
        msg.attach(part)
    
    msg.attach(MIMEText(text))
    
    server = smtplib.SMTP(host=host, port=587)
    server.login('guobingyou', 'through@pass')
    
    try:
        server.sendmail(From, To, msg.as_string())
        server.close()
    except Exception, e:
        error_msg = 'Unable to send email. Error: %s' % str(e)
        print error_msg

if __name__ == '__main__':
    send_email(file='../cf/results.txt')
