import json,os
from datetime import datetime


def read_data(filename):
    f=open(filename,"r")
    return json.load(f)

def log(logfile,logtext):
    if not os.path.isdir(os.path.dirname(logfile)):
        os.system("mkdir -p "+os.path.dirname(logfile))
    f = open(logfile,"a")
    f.write(str(datetime.now())+logtext+"\n")
    f.close()
    print(logtext)
