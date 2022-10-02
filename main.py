from collections import defaultdict
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


##### Reading messages from inbox:
#################################################

# Internet Message Access Protocol

import imaplib #, base64
import os, email

email_user = "a.vosseler@gmx.net"
email_pass = "ALEXv34osS"
IMAP_SERVER = "imap.gmx.com"

mail = imaplib.IMAP4_SSL(IMAP_SERVER)

mail.login(email_user, email_pass)

mail.select('inbox')

mail = imaplib.IMAP4_SSL(IMAP_SERVER)

mail.login(email_user, email_pass)
mail.select('inbox')


data = mail.search(None, 'ALL')
mail_ids = data[1]
id_list = mail_ids[0].split()   
first_email_id = int(id_list[0])
latest_email_id = int(id_list[-1])

for i in range(latest_email_id, first_email_id, -1):
    print("Message {}".format(i))
    data = mail.fetch(str(i), '(RFC822)' )
    for response_part in data:
      arr = response_part[0]
      if isinstance(arr, tuple):
          msg = email.message_from_string(str(arr[1],'utf-8'))


###########################################
##### Sending messages:
#################################################

#https://coderzcolumn.com/tutorials/python/smtplib-simple-guide-to-sending-mails-using-python#6


import smtplib, time, email


class EMAILS:

    def __init__(self, email_user : str = "a.vosseler@gmx.net", 
                 email_pass : str = "ALEXv34osS", 
                 SMTP_SERVER : str = "mail.gmx.net"):

        self.email_user = email_user
        self.email_pass = email_pass
        self.SMTP_SERVER = SMTP_SERVER

        ################# SMTP SSL ################################
        start = time.time()
        try:
            self.smtp_ssl = smtplib.SMTP_SSL(host=self.SMTP_SERVER, port=465)
        except Exception as e:
            print("ErrorType : {}, Error : {}".format(type(e).__name__, e))
            self.smtp_ssl = None
        
        #print("Connection Object : {}".format(self.smtp_ssl))
        print("Total Time Taken  : {:,.2f} Seconds".format(time.time() - start))

        ######### Log In to mail account ############################
        print("\nLogging In.....")  
        resp_code, response = self.smtp_ssl.login(user = self.email_user, password = self.email_pass)

        #print("Response Code : {}".format(resp_code))
        print("Response      : {}".format(response.decode()))

    def __del__(self):
        pass

    def __repr__(self):
       return "EMAIL-Class"

    def send_mail(self, input : dict = {"From" : "a.vosseler@gmx.net",     
                    "To" : ["alexandervosseler@gmail.com"],
                    "Subject" :  "Test Email"}, email_body: str = ""
        ):
 
        print("\nSending Mail..........")
        #initial = dict.fromkeys(['From', 'In', 'Subject', 'body'])
        message = email.message.EmailMessage()
        message.set_default_type("text/plain")
        message.set_content(body)
        for k,v in input.items(): message[k] = v
        response = self.smtp_ssl.send_message(msg=message)
        #print("List of Failed Recipients : {}".format(response))

        # ######### Log out to mail account ############################
        print("\nLogging Out....")
        resp_code, response = self.smtp_ssl.quit()
        #print("Response Code : {}".format(resp_code))
        print("Response      : {}".format(response.decode()))

# Run:
emails = EMAILS()    

body = '''
Hello dear colleague,

How are you doing? This is a bot generated test email.

Cheers,
'''

emails.send_mail(email_body=body)

#emails.send_mail(input=input)

#from collections import defaultdict
#from typing import List



################################

email_user = "a.vosseler@gmx.net"
email_pass = "ALEXv34osS"
SMTP_SERVER = "mail.gmx.net"

################# SMTP SSL ################################
start = time.time()
try:
    smtp_ssl = smtplib.SMTP_SSL(host=SMTP_SERVER, port=465)
except Exception as e:
    print("ErrorType : {}, Error : {}".format(type(e).__name__, e))
    smtp_ssl = None

print("Connection Object : {}".format(smtp_ssl))
print("Total Time Taken  : {:,.2f} Seconds".format(time.time() - start))

######### Log In to mail account ############################
print("\nLogging In.....")  
resp_code, response = smtp_ssl.login(user = email_user, password = email_pass)

print("Response Code : {}".format(resp_code))
print("Response      : {}".format(response.decode()))

################ Send Mail ########################
print("\nSending Mail..........")

message = email.message.EmailMessage()

message.set_default_type("text/plain")

message["From"] = "a.vosseler@gmx.net"     
message["To"] = ["alexandervosseler@gmail.com", "a.vosseler@gmx.net"]
message["Subject"] =  "Test Email"

body = '''
Hello dear colleague,

How are you doing? This is a bot generated test email.

Regards,
Alex
'''

message.set_content(body)

response = smtp_ssl.send_message(msg=message)

print("List of Failed Recipients : {}".format(response))

######### Log out to mail account ############################
print("\nLogging Out....")
resp_code, response = smtp_ssl.quit()

print("Response Code : {}".format(resp_code))
print("Response      : {}".format(response.decode()))



class emailer:

    @classmethod
    def send(cls, email_body, email_subject : str = "Test Email"):

        email_user = "a.vosseler@gmx.net"
        email_pass = "ALEXv34osS"
        SMTP_SERVER = "mail.gmx.net"

        ################# SMTP SSL ################################
        start = time.time()
        try:
            smtp_ssl = smtplib.SMTP_SSL(host=SMTP_SERVER, port=465)
        except Exception as e:
            print("ErrorType : {}, Error : {}".format(type(e).__name__, e))
            smtp_ssl = None

        print("Connection Object : {}".format(smtp_ssl))
        print("Total Time Taken  : {:,.2f} Seconds".format(time.time() - start))

        ######### Log In to mail account ############################
        print("\nLogging In.....")  
        resp_code, response = smtp_ssl.login(user = email_user, password = email_pass)

        print("Response Code : {}".format(resp_code))
        print("Response      : {}".format(response.decode()))

        ################ Send Mail ########################
        print("\nSending Mail..........")

        message = email.message.EmailMessage()

        message.set_default_type("text/plain")

        message["From"] = "a.vosseler@gmx.net"     
        message["To"] = ["alexandervosseler@gmail.com", "a.vosseler@gmx.net"]
        message["Subject"] =  email_subject

        message.set_content(email_body)

        response = smtp_ssl.send_message(msg=message)

        print("List of Failed Recipients : {}".format(response))

        ######### Log out to mail account ############################
        print("\nLogging Out....")
        resp_code, response = smtp_ssl.quit()

        print("Response Code : {}".format(resp_code))
        print("Response      : {}".format(response.decode()))


body = '''
Hello dear colleague,

How are you doing? This is a bot generated test email.

Cheers,
'''

Test = emailer.send(email_body=body, email_subject="Some Test")


