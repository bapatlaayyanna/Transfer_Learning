import datetime

def log(file_object, message):
    now = datetime.datetime.now()
    current_day_date = now.date()
    current_time = now.strftime('%H:%M:%S')
    file_object.write(str(current_day_date)+"\t"+str(current_time)+"\t"+message+"\n")